"""TEMP — contraction-metric synthesis driven by a contracting policy.

This replaces CORL. Instead of pretraining the CMG with SD-LQR controls and
freezing it, ``temp`` trains everything jointly and online:

  1. **CMG (contraction metric generator)** is trained *only* on the contraction
     (pd) condition — no SD-LQR pretrain, no c1/c2 losses. The controls/feedback
     gain (u, K = du/dx) that enter the contraction condition come from a
     **contracting policy**: an agent run at gamma -> 0. Driving the discount
     to zero makes that policy greedily minimise the *instantaneous* Riemannian
     tracking energy, so its closed loop is exactly the one the metric must
     certify. The CMG therefore learns a metric that satisfies the contraction
     condition (pd_loss -> 0) for that contracting feedback.

  2. **Contracting policy (gamma -> 0)** optimises the CMG-conditioned reward

         r(s, a) = - ||x - x_ref||_M(x)^2 / std

     i.e. the negative Riemannian energy, variance-normalised for numerical
     stability. With gamma ~ 0 this is a myopic, contraction-seeking controller.

  3. **Optimal policy (high gamma)** is *also* learned on the same
     variance-normalised Mahalanobis reward but with a high discount factor, so
     it optimises long-horizon tracking. Both policies use the same type:
     SAC (off-policy, ``OffPolicyTrainer``) or PPO (on-policy,
     ``OnPolicyTEMPTrainer``).

Why this is a valid certificate
-------------------------------
The object that certifies the closed loop is the **contraction metric M(x)
itself**, not any particular policy. The governing theorem guarantees
incremental exponential stability (IES) for *any* discount factor as long as a
valid contraction metric exists. The contracting policy's only role is to
*derive* that metric: driving its discount to zero makes it greedily minimise
the instantaneous Riemannian energy, giving a well-defined closed-loop feedback
whose contraction condition (pd_loss -> 0) pins down M. Once M is established,
the deployed high-gamma policy inherits the IES guarantee through M — it is just
as certified; the discount factor does not weaken the guarantee.

Trainer responsibility
----------------------
The replay buffer (SAC mode) and env sampling (both modes) live in the trainer,
not here. TEMP exposes three update methods that the trainer calls:

  * ``update_sac(s, a, ns, term)`` — one SAC gradient step for both cores.
  * ``update_ppo(con_batch, opt_batch, progress)`` — PPO update for both.
  * ``update_cmg()`` — one CMG gradient step.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose

from policy.base import Base
from policy.layers.policy_networks import RLActor, RLCritic
from policy.layers.sac_networks import SACActor
from policy.ppo import PPO
from policy.sac import SACCore, move_optimizer_states


class TEMP(Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        state_dim: int,
        CMG: nn.Module,
        get_f_and_B,
        data: dict,
        action_scale=1.0,
        action_bias=0.0,
        # network sizes
        actor_dim: list = (256, 256),
        critic_dim: list = (256, 256),
        actor_activation: str = "relu",
        # which policy type to use (both contracting AND optimal share the type)
        optimal_policy: str = "sac",  # {"sac", "ppo"}
        # if True, only contracting policy is trained (no optimal actor)
        con_only: bool = False,
        # discount factors
        gamma_contracting: float = 0.0,
        gamma_optimal: float = 0.99,
        # SAC hyperparameters (used when optimal_policy == "sac")
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        autotune_alpha: bool = True,
        # CMG / contraction
        W_lr: float = 3e-4,
        w_ub: float = 10.0,
        w_lb: float = 0.1,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        W_entropy_scaler: float = 1e-3,
        cmg_minibatch_size: int = 1024,
        cmg_updates_per_iter: int = 50,
        reward_norm_beta: float = 0.99,
        # reward shaping
        tracking_scaler: float = 1.0,
        control_scaler: float = 0.0,
        # PPO params (used when optimal_policy == "ppo")
        eps_clip: float = 0.2,
        K: int = 5,
        gae: float = 0.95,
        target_kl: float = 0.03,
        entropy_scaler: float = 1e-3,
        num_minibatch: int = 4,
        minibatch_size: int = 1024,
        num_windows: int = 1,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = "TEMP-Con" if con_only else "TEMP"
        self.con_only = con_only
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.state_dim = state_dim
        self.num_windows = num_windows

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            self.get_f_and_B.eval()
        self.data = data

        self.CMG = CMG
        self.w_ub = w_ub
        self.w_lb = w_lb
        self.lbd = lbd
        self.eps = eps
        self.W_entropy_scaler = W_entropy_scaler
        self.cmg_minibatch_size = cmg_minibatch_size
        self.cmg_updates_per_iter = cmg_updates_per_iter
        self.tracking_scaler = tracking_scaler
        self.control_scaler = control_scaler

        # IES follows from the contraction metric M for any discount factor.
        # gamma_optimal is kept for logging / eval context only.
        self.gamma = gamma_optimal
        self.gamma_contracting = gamma_contracting
        self.gamma_optimal = gamma_optimal

        self.optimal_policy = optimal_policy

        # ------------------------------------------------------------------ #
        # Build contracting + optimal policies
        # ------------------------------------------------------------------ #
        if optimal_policy == "sac":
            self.con_actor = SACActor(
                x_dim, u_dim, state_dim, list(actor_dim),
                action_scale=action_scale, action_bias=action_bias,
                num_windows=num_windows, activation=actor_activation,
            )
            self.con_core = SACCore(
                state_dim, u_dim, self.con_actor, list(critic_dim),
                gamma=gamma_contracting, tau=tau, actor_lr=actor_lr,
                critic_lr=critic_lr, alpha_lr=alpha_lr, init_alpha=init_alpha,
                autotune_alpha=autotune_alpha, device=device,
            )
            if not con_only:
                self.opt_actor = SACActor(
                    x_dim, u_dim, state_dim, list(actor_dim),
                    action_scale=action_scale, action_bias=action_bias,
                    num_windows=num_windows, activation=actor_activation,
                )
                self.opt_core = SACCore(
                    state_dim, u_dim, self.opt_actor, list(critic_dim),
                    gamma=gamma_optimal, tau=tau, actor_lr=actor_lr,
                    critic_lr=critic_lr, alpha_lr=alpha_lr, init_alpha=init_alpha,
                    autotune_alpha=autotune_alpha, device=device,
                )
            else:
                self.opt_actor = None
                self.opt_core = None
            self.con_ppo = None
            self.opt_ppo = None

        elif optimal_policy == "ppo":
            self.con_actor = RLActor(
                x_dim=x_dim, u_dim=u_dim, hidden_dim=list(actor_dim),
                mode="stochastic", activation=actor_activation,
            )
            con_critic = RLCritic(state_dim, hidden_dim=list(critic_dim))
            self.con_ppo = PPO(
                x_dim=x_dim, u_dim=u_dim, latent_dim=x_dim,
                num_windows=num_windows, actor=self.con_actor, critic=con_critic,
                actor_lr=actor_lr, critic_lr=critic_lr,
                num_minibatch=num_minibatch, minibatch_size=minibatch_size,
                eps_clip=eps_clip, entropy_scaler=entropy_scaler,
                target_kl=target_kl, gamma=gamma_contracting, gae=gae, K=K,
                nupdates=nupdates, device=device,
            )
            if not con_only:
                self.opt_actor = RLActor(
                    x_dim=x_dim, u_dim=u_dim, hidden_dim=list(actor_dim),
                    mode="stochastic", activation=actor_activation,
                )
                opt_critic = RLCritic(state_dim, hidden_dim=list(critic_dim))
                self.opt_ppo = PPO(
                    x_dim=x_dim, u_dim=u_dim, latent_dim=x_dim,
                    num_windows=num_windows, actor=self.opt_actor, critic=opt_critic,
                    actor_lr=actor_lr, critic_lr=critic_lr,
                    num_minibatch=num_minibatch, minibatch_size=minibatch_size,
                    eps_clip=eps_clip, entropy_scaler=entropy_scaler,
                    target_kl=target_kl, gamma=gamma_optimal, gae=gae, K=K,
                    nupdates=nupdates, device=device,
                )
            else:
                self.opt_actor = None
                self.opt_ppo = None
            self.con_core = None
            self.opt_core = None
        else:
            raise ValueError(f"Unknown optimal_policy: {optimal_policy!r}")

        # con_only deploys the contracting actor; full TEMP deploys the optimal actor.
        # Trainer swaps self.actor to con_actor temporarily for contracting rollouts.
        self.actor = self.con_actor if con_only else self.opt_actor

        # CMG optimizer (pd-loss only).
        self.W_optimizer = torch.optim.Adam(self.CMG.parameters(), lr=W_lr)

        # Running EMA of the variance of the Riemannian energy (reward normaliser).
        self.reward_norm_beta = reward_norm_beta
        self.running_reward_var = None

        self.to(self._dtype).to(self.device)

    # ------------------------------------------------------------------ #
    # device handling
    # ------------------------------------------------------------------ #
    def to_device(self, device):
        super().to_device(device)
        move_optimizer_states(self, device)

    # ------------------------------------------------------------------ #
    # deployment / sampler interface
    # ------------------------------------------------------------------ #
    def _deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        if hasattr(self.actor, "mean_control"):
            return self.actor.mean_control(state)
        prev = self.actor.mode
        self.actor.mode = "deterministic"
        a, _ = self.actor(state)
        self.actor.mode = prev
        return a

    def forward(self, state: np.ndarray):
        """Uses ``self.actor`` (opt_actor by default; trainer may swap temporarily)."""
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if self.training:
                a, md = self.actor(state)
                metaData = {
                    "probs": md["probs"],
                    "logprobs": md["logprobs"],
                    "entropy": md["entropy"],
                }
            else:
                a = self._deterministic_action(state)
                z = torch.zeros(a.shape[0], 1, device=self.device)
                metaData = {"probs": torch.ones_like(z), "logprobs": z, "entropy": z}
        return a, metaData

    # ------------------------------------------------------------------ #
    # contracting actor: deterministic mean, differentiable in state
    # ------------------------------------------------------------------ #
    def _con_actor_mean(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic control from the contracting actor (differentiable in state).

        Works for both SACActor (has ``mean_control``) and RLActor (uses
        deterministic forward without sampling noise).
        """
        if hasattr(self.con_actor, "mean_control"):
            return self.con_actor.mean_control(state)
        prev = self.con_actor.mode
        self.con_actor.mode = "deterministic"
        u, _ = self.con_actor(state)
        self.con_actor.mode = prev
        return u

    # ------------------------------------------------------------------ #
    # CMG-conditioned, variance-normalised Riemannian reward
    # ------------------------------------------------------------------ #
    def compute_mahalanobis_reward(
        self, states: torch.Tensor, controls: torch.Tensor = None
    ) -> torch.Tensor:
        x = states[:, : self.x_dim]
        xref = states[:, self.x_dim : 2 * self.x_dim]
        e = (x - xref).unsqueeze(-1)

        with torch.no_grad():
            raw_W, _ = self.CMG(x, deterministic=True)
            W = self._bound_W(raw_W)
            I = torch.eye(self.x_dim, device=self.device, dtype=W.dtype)
            M = torch.linalg.solve(W, I.unsqueeze(0).expand_as(W))
            quad = (transpose(e, 1, 2) @ M @ e).squeeze(-1)

            batch_var = quad.var(unbiased=False).item()
            if self.running_reward_var is None:
                self.running_reward_var = batch_var
            else:
                self.running_reward_var = (
                    self.reward_norm_beta * self.running_reward_var
                    + (1.0 - self.reward_norm_beta) * batch_var
                )
            std = (self.running_reward_var ** 0.5) + 1e-8

            reward = -self.tracking_scaler * quad / std
            if self.control_scaler > 0 and controls is not None:
                reward = reward - self.control_scaler * controls.pow(2).sum(
                    -1, keepdim=True
                )
        return reward

    # ------------------------------------------------------------------ #
    # CMG contraction (pd) loss
    # ------------------------------------------------------------------ #
    def compute_cmg_loss(self):
        I = torch.eye(self.x_dim, device=self.device)

        buffer_size = self.data["x"].shape[0]
        n = min(self.cmg_minibatch_size, buffer_size)
        idx = np.random.choice(buffer_size, size=n, replace=False)
        x = self.to_tensor(self.data["x"][idx]).requires_grad_()
        xref = self.to_tensor(self.data["xref"][idx])
        uref = self.to_tensor(self.data["uref"][idx])

        state = torch.cat([x, xref, uref], dim=1)
        u = self._con_actor_mean(state)
        Kfb = self.Jacobian(u, x)
        u = u.detach()
        Kfb = Kfb.detach()

        raw_W, info_W = self.CMG(x)
        W = self._bound_W(raw_W)
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        f, B, _ = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)
        B = B.to(self._dtype).to(self.device)

        DfDx = self.Jacobian(f, x)
        DBDx = self.B_Jacobian(B, x)
        f = f.detach()
        B = B.detach()

        A = DfDx + sum(
            u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
            for i in range(self.u_dim)
        )
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        ABK = A + matmul(B, Kfb)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)

        pd_loss, pd_reg = self.loss_pos_matrix_eigen(-Cu)
        entropy_loss = info_W["entropy"].mean() * self.W_entropy_scaler
        loss = pd_loss + pd_reg - entropy_loss

        with torch.no_grad():
            cu_eig = torch.linalg.eigvalsh(
                Cu.cpu() if Cu.device.type == "mps" else Cu
            )
            cu_max_eig = cu_eig.max(dim=-1).values.mean().item()

        return loss, {
            "pd_loss": pd_loss.item(),
            "pd_reg": pd_reg.item(),
            "entropy_loss": entropy_loss.item(),
            "cu_max_eig": cu_max_eig,
        }

    def _optimize_cmg(self, loss: torch.Tensor) -> float:
        self.W_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.CMG.parameters(), max_norm=10.0)
        gd = self.compute_gradient_norm(
            [self.CMG], ["CMG"], dir=self.name, device=self.device
        )
        self.W_optimizer.step()
        return gd.get(f"{self.name}/grad/CMG", 0.0)

    # ------------------------------------------------------------------ #
    # Update methods called by the trainer
    # ------------------------------------------------------------------ #
    def update_sac(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        next_states: torch.Tensor,
        terminations: torch.Tensor,
    ) -> dict:
        """One SAC gradient step for both contracting (γ→0) and optimal (high γ) cores.

        The env reward stored in the buffer is intentionally ignored; we
        recompute the Mahalanobis reward from the current M each call so the
        Q-function targets stay consistent as the metric evolves.
        """
        self.train()
        r = self.compute_mahalanobis_reward(states, controls)

        con_info = self.con_core.update(states, controls, next_states, r, terminations)
        loss_dict = {f"{self.name}/contracting/{k}": v for k, v in con_info.items()}
        if not self.con_only:
            opt_info = self.opt_core.update(states, controls, next_states, r, terminations)
            for k, v in opt_info.items():
                loss_dict[f"{self.name}/optimal/{k}"] = v
        loss_dict[f"{self.name}/RL_analytics/reward_std"] = (
            (self.running_reward_var or 0.0) ** 0.5
        )
        return loss_dict

    def update_ppo(self, con_batch: dict, opt_batch: dict = None, progress: float = 0.0) -> dict:
        """PPO update for both contracting and optimal policies.

        Mahalanobis rewards are recomputed from the current M and injected into
        the batches before calling each PPO's ``learn()`` method.
        """
        self.train()

        con_states = self.to_tensor(con_batch["states"])
        con_r = self.compute_mahalanobis_reward(con_states)
        con_batch = dict(con_batch, rewards=con_r.detach().cpu().numpy().astype(np.float32))
        con_loss, _, _ = self.con_ppo.learn(con_batch, progress)
        loss_dict = {f"{self.name}/contracting/{k}": v for k, v in con_loss.items()}

        if not self.con_only and opt_batch is not None:
            opt_states = self.to_tensor(opt_batch["states"])
            opt_r = self.compute_mahalanobis_reward(opt_states)
            opt_batch = dict(opt_batch, rewards=opt_r.detach().cpu().numpy().astype(np.float32))
            opt_loss, _, _ = self.opt_ppo.learn(opt_batch, progress)
            for k, v in opt_loss.items():
                loss_dict[f"{self.name}/optimal/{k}"] = v

        loss_dict[f"{self.name}/RL_analytics/reward_std"] = (
            (self.running_reward_var or 0.0) ** 0.5
        )
        return loss_dict

    def learn(self, batch=None, progress: float = 0.0):
        """Satisfies the Base abstract interface; TEMP delegates to the trainer."""
        raise NotImplementedError(
            "TEMP does not use learn(). "
            "Use update_sac / update_ppo / update_cmg via OffPolicyTrainer or OnPolicyTEMPTrainer."
        )

    def update_cmg(self) -> dict:
        """One CMG gradient step. Returns a loss-info dict for logging."""
        self.train()
        loss, info = self.compute_cmg_loss()
        grad_norm = self._optimize_cmg(loss)
        return {
            f"{self.name}/CMG/pd_loss": info["pd_loss"],
            f"{self.name}/CMG/pd_reg": info["pd_reg"],
            f"{self.name}/CMG/entropy_loss": info["entropy_loss"],
            f"{self.name}/CMG/cu_max_eig": info["cu_max_eig"],
            f"{self.name}/CMG/grad_norm": grad_norm,
        }
