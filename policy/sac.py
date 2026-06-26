"""Soft Actor-Critic (SAC).

Two layers:

  * :class:`SACCore` — a self-contained SAC learner (tanh-squashed actor, twin Q
    critics with Polyak-averaged targets, optional auto-tuned entropy temperature
    alpha). One ``update()`` call performs a single SAC gradient step on a given
    minibatch. It is reused as a *component* by the ``temp`` algorithm, which runs
    two cores at different discount factors against a shared replay buffer.
  * :class:`SAC` — a standalone :class:`Base` policy that wraps one core plus a
    :class:`ReplayBuffer`, so SAC can be trained directly via the existing
    :class:`OnlineTrainer` (algo name ``sac``).

The on-policy sampler hands ``learn`` a fresh on-policy *batch* each iteration;
SAC simply appends those transitions to its replay buffer and then performs
``utd * n_new`` off-policy gradient steps.
"""

import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.base import Base
from policy.layers.sac_networks import QCritic, SACActor
from utils.replay_buffer import ReplayBuffer


def move_optimizer_states(module: nn.Module, device):
    """Move the state tensors of every optimizer held by ``module`` or its
    submodules onto ``device``.

    :meth:`Base.to_device` only scans the top-level policy's ``__dict__``; SAC
    keeps its optimizers inside :class:`SACCore` submodules, so we recurse here.
    Called by the overridden ``to_device`` after the sampler ships the policy
    to CPU workers and back.
    """
    for m in module.modules():
        for v in m.__dict__.values():
            if isinstance(v, torch.optim.Optimizer):
                for state in v.state.values():
                    for k, t in state.items():
                        if isinstance(t, torch.Tensor):
                            state[k] = t.to(device)


class SACCore(nn.Module):
    """A single SAC learner (actor + twin critics + temperature)."""

    def __init__(
        self,
        state_dim: int,
        u_dim: int,
        actor: SACActor,
        critic_dim: list,
        gamma: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float = None,
        lr_decay_lambda=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.u_dim = u_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = actor
        self.q1 = QCritic(state_dim, u_dim, critic_dim)
        self.q2 = QCritic(state_dim, u_dim, critic_dim)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        for p in self.q1_targ.parameters():
            p.requires_grad_(False)
        for p in self.q2_targ.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )

        if lr_decay_lambda is not None:
            from torch.optim.lr_scheduler import LambdaLR
            self.actor_lr_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lr_decay_lambda)
            self.critic_lr_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lr_decay_lambda)
        else:
            self.actor_lr_scheduler = None
            self.critic_lr_scheduler = None

        # Entropy temperature: auto-tuned via the dual of the entropy constraint
        # E[-logπ] >= target_entropy (default -u_dim, the SAC heuristic).
        self.autotune_alpha = autotune_alpha
        self.target_entropy = (
            float(-u_dim) if target_entropy is None else float(target_entropy)
        )
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha))), requires_grad=autotune_alpha
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr) if autotune_alpha else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def soft_update(self):
        with torch.no_grad():
            for src, tgt in (
                (self.q1, self.q1_targ),
                (self.q2, self.q2_targ),
            ):
                for p, pt in zip(src.parameters(), tgt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
    ) -> dict:
        alpha = self.alpha.detach()

        # === critic target === #
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_states)
            q1_t = self.q1_targ(next_states, next_a)
            q2_t = self.q2_targ(next_states, next_a)
            min_q_t = torch.min(q1_t, q2_t) - alpha * next_logp
            target = rewards + self.gamma * (1.0 - terminations) * min_q_t

        q1 = self.q1(states, controls)
        q2 = self.q2(states, controls)
        critic_loss = 0.5 * (
            torch.nn.functional.mse_loss(q1, target)
            + torch.nn.functional.mse_loss(q2, target)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=10.0
        )
        self.critic_optimizer.step()

        # === actor === #
        a, logp = self.actor.sample(states)
        q1_pi = self.q1(states, a)
        q2_pi = self.q2(states, a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # === temperature === #
        if self.autotune_alpha:
            alpha_loss = -(
                self.log_alpha * (logp + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0

        self.soft_update()

        if getattr(self, "actor_lr_scheduler", None) is not None:
            self.actor_lr_scheduler.step()
        if getattr(self, "critic_lr_scheduler", None) is not None:
            self.critic_lr_scheduler.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha.item(),
            "entropy": (-logp).mean().item(),
            "q_value": min_q_pi.mean().item(),
        }


class SAC(Base):
    """Standalone Soft Actor-Critic policy (off-policy, replay-buffered)."""

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        state_dim: int,
        actor: SACActor,
        critic_dim: list,
        gamma: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float = None,
        buffer_size: int = 1_000_000,
        sac_batch_size: int = 256,
        utd_ratio: float = 1.0,
        learning_starts: int = 5000,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = "SAC"
        self.device = device
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.sac_batch_size = sac_batch_size
        self.utd_ratio = utd_ratio
        self.learning_starts = learning_starts

        self.actor = actor
        self.core = SACCore(
            state_dim=state_dim,
            u_dim=u_dim,
            actor=actor,
            critic_dim=critic_dim,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            autotune_alpha=autotune_alpha,
            target_entropy=target_entropy,
            lr_decay_lambda=self.lr_decay_lambda,
            device=device,
        )

        self.buffer = ReplayBuffer(
            state_dim=state_dim,
            u_dim=u_dim,
            buffer_size=buffer_size,
            batch_size=sac_batch_size,
            device=device,
        )

        self.to(self._dtype).to(self.device)

    def to_device(self, device):
        super().to_device(device)
        self.buffer.device = device
        move_optimizer_states(self, device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # Deterministic (mean) action whenever the policy is in eval mode; the
        # trainer sets train() before sampling and eval() before evaluation.
        deterministic = deterministic or (not self.training)
        with torch.no_grad():
            if deterministic:
                a = self.actor.mean_control(state)
                metaData = {
                    "probs": torch.ones(a.shape[0], 1, device=self.device),
                    "logprobs": torch.zeros(a.shape[0], 1, device=self.device),
                    "entropy": torch.zeros(a.shape[0], 1, device=self.device),
                }
            else:
                a, md = self.actor(state)
                metaData = {
                    "probs": md["probs"],
                    "logprobs": md["logprobs"],
                    "entropy": md["entropy"],
                }
        return a, metaData

    def _append_batch(self, batch: dict):
        self.buffer.direct_append(
            states=batch["states"],
            controls=batch["controls"],
            next_states=batch["next_states"],
            rewards=batch["rewards"],
            terminations=batch["terminations"],
        )

    def learn(self, batch: dict, progress: float):
        self.train()
        t0 = time.time()
        self.progress = progress

        self._append_batch(batch)
        n_new = batch["states"].shape[0]

        if self.buffer.size < self.learning_starts:
            # Warmup: fill the buffer without a policy update. Report zero updates
            # so the trainer neither counts nor logs this as a training step; the
            # whole warmup collapses into a single wandb logging tick.
            loss_dict = {
                f"{self.name}/RL_analytics/buffer_size": self.buffer.size,
                f"{self.name}/RL_analytics/n_updates": 0,
            }
            self.eval()
            return loss_dict, {}, time.time() - t0

        n_updates = max(1, int(self.utd_ratio * n_new))
        agg = {}
        for _ in range(n_updates):
            s, c, ns, r, term = self.buffer.sample()
            info = self.core.update(s, c, ns, r, term)
            for k, v in info.items():
                agg.setdefault(k, []).append(v)

        loss_dict = {
            f"{self.name}/RL_loss/{k}": float(np.mean(v)) for k, v in agg.items()
        }
        loss_dict[f"{self.name}/RL_analytics/buffer_size"] = self.buffer.size
        loss_dict[f"{self.name}/RL_analytics/n_updates"] = n_updates
        loss_dict[f"{self.name}/RL_analytics/avg_rewards"] = float(
            np.mean(batch["rewards"])
        )

        self.eval()
        return loss_dict, {}, time.time() - t0
