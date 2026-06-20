import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from policy.base import Base
from utils.functions import estimate_advantages


class CARL(Base):
    def __init__(
        self,
        # Learning parameters
        x_dim: int,
        u_dim: int,
        data: dict,
        CMG: nn.Module,
        get_f_and_B: Callable,
        actor: nn.Module,
        critic: nn.Module,
        W_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        # CMG parameters
        disable_CMG_training: bool = False,
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        W_entropy_scaler: float = 1e-3,
        W_patience: int = 20,  # Number of updates to wait before stopping
        reward_mode: str = "default",
        # TRPO parameters
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        target_kl: float = 0.03,
        # PPO parameters
        eps_clip: float = 0.2,
        K: int = 5,
        entropy_scaler: float = 1e-3,
        # RL parameters
        num_windows: int = 1,
        gamma: float = 0.99,
        gae: float = 0.95,
        l2_reg: float = 1e-8,
        tracking_scaler: float = 1.0,
        control_scaler: float = 0.0,
        nupdates: int = 1,
        policy_updates_per_cmg_update: int = 50,
        warmup_epochs: int = 10000,
        device: str = "cpu",
    ):
        super().__init__()

        # constants
        self.name = "Algorithm"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_windows = num_windows

        self.data = data
        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.W_entropy_scaler = W_entropy_scaler
        self.W_patience = W_patience
        self.entropy_scaler = entropy_scaler
        self.tracking_scaler = tracking_scaler
        self.control_scaler = control_scaler
        self.eps = eps
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.lbd = lbd
        self.damping = damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.w_ub = w_ub
        self.w_lb = w_lb

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

        self.nupdates = nupdates
        self.policy_updates_per_cmg_update = policy_updates_per_cmg_update
        self.num_W_updates = 1
        self.num_RL_updates = 1

        self.W_patience_counter = 0
        self.best_W_loss = float("inf")
        self.stop_W_training = False  # Flag to freeze W updates

        # trainable networks
        self.CMG = CMG
        self.disable_CMG_training = disable_CMG_training
        self.actor = actor
        self.critic = critic

        self.W_optimizer = torch.optim.Adam(
            [
                {"params": self.CMG.parameters(), "lr": W_lr},
            ]
        )

        self.RL_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr, "name": "actor"},
                {"params": self.critic.parameters(), "lr": critic_lr, "name": "critic"},
            ]
        )

        self.progress = 0.0
        self.RL_lr_scheduler = LambdaLR(
            self.RL_optimizer, lr_lambda=self.timestep_lr_lambda
        )
        self.W_lr_scheduler = LambdaLR(
            self.W_optimizer, lr_lambda=self.timestep_lr_lambda
        )

        self.to(self._dtype).to(self.device)

        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            self.warmup_W()

    def timestep_lr_lambda(self, _):
        """
        Exponential decay: Multiplier = exp(-k * progress)
        """
        # Controls how fast it drops.
        # k=10.0 means it drops to ~0.0045% (exp(-10)) by the end.
        # k=5.0 means it drops to ~0.6% (exp(-5)) by the end.
        # k=3.0 means it drops to ~5.0% (exp(-3)) by the end.
        # decay_k = 3.0
        # return math.exp(-decay_k * self.progress)

        return max(0.0, 1.0 - self.progress)

    def anneal_entropy_scaler(self):
        """Linearly anneal the entropy scaler from initial to final value based on progress."""
        decay_k = 10.0
        return self.entropy_scaler * math.exp(-decay_k * self.progress)



    def forward(self, state: np.ndarray):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        a, metaData = self.actor(state)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def compute_W_loss(self, warming_up: bool = False):
        #
        I = torch.eye(self.x_dim, device=self.device)

        # === SAMPLE BATCH === #
        batch = dict()
        buffer_size, batch_size = self.data["x"].shape[0], 1024
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        for key in self.data.keys():
            # Sample a batch of 1024
            batch[key] = self.data[key][indices]

        # === PREPARE TENSORS === #
        x = self.to_tensor(batch["x"]).requires_grad_().repeat(4, 1)
        xref = self.to_tensor(batch["xref"]).repeat(4, 1)
        uref = self.to_tensor(batch["uref"]).repeat(4, 1)

        # since online we do not do below
        state = torch.concatenate([x, xref, uref], dim=1)
        u, _ = self.actor(state)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        # detach actor gradients
        u = u.detach()
        K = K.detach()

        raw_W, info_W = self.CMG(x)  # n, x_dim, x_dim
        # Add lower-bound scaled identity to guarantee positive definiteness
        W = raw_W + self.w_lb * I
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        # === GET DYNAMICS === #
        f, B, Bbot = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)  # n, x_dim
        B = B.to(self._dtype).to(self.device)  # n, x_dim, action
        Bbot = Bbot.to(self._dtype).to(self.device)  #

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim

        f = f.detach()
        B = B.detach()
        Bbot = Bbot.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.u_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        # # === VERIFY DERIVATIVES (K, DfDx, DBDx, dot_M) are NONZERO === #
        # print(
        #     "Derivative norms - K: {:.4e}, DfDx: {:.4e}, DBDx: {:.4e}, dot_M: {:.4e}".format(
        #         torch.norm(K).item(),
        #         torch.norm(DfDx).item(),
        #         torch.norm(DBDx).item(),
        #         torch.norm(dot_M).item(),
        #     )
        # )
        # if torch.norm(K).item() < 1e-6:
        #     print("Warning: K is zero!")
        # if torch.norm(DfDx).item() < 1e-6:
        #     print("Warning: DfDx is zero!")
        # if torch.norm(DBDx).item() < 1e-6:
        #     print("Warning: DBDx is zero!")
        # if torch.norm(dot_M).item() < 1e-6:
        #     print("Warning: dot_M is zero!")

        # contraction condition
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # this has to be a negative definite matrix
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.u_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
            C2_inner = DbW - 2 * sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)

            C2_inners.append(C2_inner)
            C2s.append(C2)

        ### DEFINE PD MATRICES ###
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)
        C1 = C1 + self.eps * torch.eye(C1.shape[-1], device=self.device)
        C2 = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        overshoot = W - self.w_ub * I

        # === DEFINE LOSSES === #
        pd_loss, pd_reg = self.loss_pos_matrix_eigen(-Cu)
        c1_loss, c1_reg = self.loss_pos_matrix_eigen(-C1)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_eigen(-overshoot)
        c2_loss = C2

        entropy_loss = info_W["entropy"].mean() * self.W_entropy_scaler

        if warming_up:
            loss = c1_loss + c2_loss + c1_reg + overshoot_loss + overshoot_reg
        else:
            self.record_eigenvalues(Cu, dot_M, sym_MABK, C1, C2, overshoot)
            loss = (
                overshoot_loss
                + pd_loss
                + c1_loss
                + c2_loss
                + pd_reg
                + c1_reg
                + overshoot_reg
                - entropy_loss
            )

        return (
            loss,
            {
                "pd_loss": pd_loss,
                "c1_loss": c1_loss,
                "c2_loss": c2_loss,
                "overshoot_loss": overshoot_loss,
                "entropy_loss": entropy_loss,
            },
        )

    def optimize_W_params(self, loss: torch.Tensor):
        # === OPTIMIZATION STEP === #
        self.W_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.CMG.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.CMG],
            ["CMG"],
            dir="CARL",
            device=self.device,
        )
        self.W_optimizer.step()

        return grad_dict

    def warmup_W(self):
        best_loss, stagnant_epochs = float("inf"), 0
        self.define_loss_lists()

        with tqdm(range(self.warmup_epochs), desc="Warmup Phase") as pbar:
            for epoch in pbar:
                # 1. Train Step
                loss, infos = self.compute_W_loss(warming_up=True)
                self.optimize_W_params(loss)

                # 2. Get scalar loss
                current_loss = loss.item() if hasattr(loss, "item") else loss

                # 3. Update Progress Bar
                pbar.set_postfix(loss=f"{current_loss:.4f}")
                self.save_values_to_loss_lists(
                    loss.item(),
                    0.0,
                    infos["pd_loss"].item(),
                    infos["c1_loss"].item(),
                    infos["c2_loss"].item(),
                    infos["overshoot_loss"].item(),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )

                # 4. Convergence Check (Minimal Change)
                # We check if improvement is positive but very small
                if current_loss >= best_loss:
                    stagnant_epochs += 1
                else:
                    stagnant_epochs = 0  # Reset if we see good improvement or a spike
                    best_loss = current_loss

                # Check if we have been stagnant for 'self.W_patience' epochs
                if stagnant_epochs >= self.W_patience:
                    pbar.write(
                        f"✓ Warmup converged: Loss stabilized at {current_loss:.4f} for {self.W_patience} epochs)"
                    )
                    break

            else:
                pbar.write(
                    f"⚠ Max warmup epochs ({self.warmup_epochs}) reached without full stabilization."
                )

        fig = self.plot_warmup_result()
        self.warmup_result = fig

    def learn(self, batch: dict, progress: float):
        self.progress = progress

        loss_dict, supp_dict = {}, {}

        # Implement the freeze-and-learn scheme here
        W_update_time = 0
        if (
            self.num_RL_updates % self.policy_updates_per_cmg_update == 0
            and not self.stop_W_training
            and not self.disable_CMG_training
        ):
            # pass
            W_loss_dict, W_supp_dict, W_update_time = self.learn_W()
            loss_dict.update(W_loss_dict)
            supp_dict.update(W_supp_dict)

        RL_loss_dict, RL_supp_dict, RL_update_time = self.learn_ppo(batch)

        loss_dict.update(RL_loss_dict)
        supp_dict.update(RL_supp_dict)

        self.W_lr_scheduler.step()
        self.RL_lr_scheduler.step()

        self.actor.anneal_stddev(progress, mode="exponential")

        update_time = W_update_time + RL_update_time
        self.num_RL_updates += 1

        return loss_dict, supp_dict, update_time

    def learn_W(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # 1. Compute Loss (We always compute this to monitor, even if frozen)
        loss, infos = self.compute_W_loss()
        current_loss = loss.item()

        # 2. Check for significant improvement
        if current_loss < self.best_W_loss:
            self.best_W_loss = current_loss
            self.W_patience_counter = 0
        else:
            self.W_patience_counter += 1

        if self.W_patience_counter >= self.W_patience:
            self.stop_W_training = True
            print(
                f"[{self.name}] W network loss stabilized. Stopping updates (Best: {self.best_W_loss:.5f})."
            )

        # 3. Optimize (Only if not stopped)
        grad_dict = self.optimize_W_params(loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_W_updates % 10 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict[f"{self.name}/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/CMG_loss/loss": current_loss,
            f"{self.name}/CMG_loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/CMG_loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/CMG_loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/CMG_loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/CMG_loss/entropy_loss": infos["entropy_loss"].item(),
            f"{self.name}/CMG_analytics/lbd": self.lbd,
            f"{self.name}/lr/W_lr": self.W_lr_scheduler.get_last_lr()[0],
            f"{self.name}/CMG_analytics/W_frozen": float(
                self.stop_W_training
            ),  # Log status
        }

        norm_dict = self.compute_weight_norm(
            [self.CMG],
            ["CMG"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        self.eval()
        update_time = time.time() - t0
        self.num_W_updates += 1

        return loss_dict, supp_dict, update_time

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.to_tensor(batch["states"])
        controls = self.to_tensor(batch["controls"])
        original_rewards = self.to_tensor(batch["rewards"])
        rewards, reward_ratio = self.get_rewards(states, controls)
        terminations = self.to_tensor(batch["terminations"])
        old_logprobs = self.to_tensor(batch["logprobs"])

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses, actor_losses, value_losses, entropy_losses = [], [], [], []
        clip_frcontrols, target_kl, grad_dicts = [], [], []

        # Tracking KL across the entire update for the scheduler
        total_kl = 0.0
        num_updates = 0

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_controls = states[indices], controls[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]

                # 1. Critic Loss (with optional regularization)
                value_loss = self.critic_loss(mb_states, mb_returns)

                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
                    mb_states, mb_controls, mb_old_logprobs, mb_advantages
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_frcontrols.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Accumulate KL for scheduler logic
                total_kl += kl_div.item()
                num_updates += 1

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss
                losses.append(loss.item())

                # Update parameters
                self.RL_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.RL_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        supp_dict = {}
        loss_dict = {
            f"{self.name}/RL_loss/loss": np.mean(losses),
            f"{self.name}/RL_loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/RL_loss/value_loss": np.mean(value_losses),
            f"{self.name}/RL_loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/lr/actor_lr": self.RL_optimizer.param_groups[0]["lr"],
            f"{self.name}/RL_analytics/std_dev": self.actor.logstd.exp().mean().item(),
            f"{self.name}/RL_analytics/clip_fraction": np.mean(clip_frcontrols),
            f"{self.name}/RL_analytics/kl_divergence": (
                sum(target_kl) / len(target_kl) if target_kl else 0.0
            ),
            f"{self.name}/RL_analytics/K-epoch": k + 1,
            f"{self.name}/RL_analytics/avg_rewards": torch.mean(
                original_rewards
            ).item(),
            f"{self.name}/RL_analytics/corrected_avg_rewards": torch.mean(
                rewards
            ).item(),
            f"{self.name}/RL_analytics/reward_ratio": reward_ratio,
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, controls, rewards, terminations, old_logprobs
        self.eval()

        update_time = time.time() - t0
        return loss_dict, supp_dict, update_time

    def get_rewards(self, states: torch.Tensor, controls: torch.Tensor):
        # Helper to check and report NaNs
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"DEBUG: NaN detected in '{name}'")
                if name == "W":
                    # Check if matrix is ill-conditioned
                    cond = torch.linalg.cond(tensor)
                    print(
                        f"  -> W condition number: max={cond.max().item()}, min={cond.min().item()}"
                    )
                return True
            return False

        x, xref, uref = self.actor.trim_state(states)
        tracking_error = (x - xref).unsqueeze(-1)
        control_effort = torch.linalg.norm(controls, dim=-1, keepdim=True)

        check_nan(states, "input states")
        check_nan(controls, "input controls")

        with torch.no_grad():
            W, _ = self.CMG(x, deterministic=True)
            W += self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )

            if check_nan(W, "W"):
                # Optional: print W values to see if they are inf or zero
                print(f"  -> W mean: {W.mean().item()}")

            M = torch.linalg.solve(W, torch.eye(W.shape[-1], device=W.device, dtype=W.dtype).unsqueeze(0).expand_as(W))
            check_nan(M, "M (after inverse)")

            tracking_errorT = transpose(tracking_error, 1, 2)

            # Matrix multiplication check
            inner_quad = (tracking_errorT @ M @ tracking_error).squeeze(-1)
            tracking_reward = -self.tracking_scaler * inner_quad
            control_reward = -self.control_scaler * control_effort

        if self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + abs(tracking_reward))
            control_reward = 1 / (1 + abs(control_reward))
            check_nan(tracking_reward, "tracking_reward (inverse mode)")

        rewards = (0.5 * tracking_reward) + (0.5 * control_reward)

        if check_nan(rewards, "final rewards"):
            # If rewards are NaN, the tuple return might fail on .item()
            print(f"  -> tracking_reward: {tracking_reward.mean().item()}")
            print(f"  -> control_reward: {control_reward.mean().item()}")

        # Safe return for the ratio to avoid .item() crash on NaN
        ratio = (tracking_reward.mean() / (control_reward.mean(0) + 1e-8)).item()

        return rewards, ratio

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_controls: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = self.actor(mb_states)
        logprobs = self.actor.log_prob(metaData["dist"], mb_controls)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        entropy_scaler = self.anneal_entropy_scaler()
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(self, mb_states: torch.Tensor, mb_returns: torch.Tensor):
        mb_values = self.critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)
        return value_loss
