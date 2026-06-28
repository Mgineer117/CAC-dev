import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# from actor.layers.building_blocks import MLP
from policy.base import Base

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.functions import estimate_advantages

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class PPO(Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        latent_dim: int,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super(PPO, self).__init__()

        # constants
        self.name = "Algorithm"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = actor.u_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip

        self.nupdates = nupdates

        # trainable networks
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.progress = 0.0
        self.ppo_lr_scheduler = LambdaLR(
            self.optimizer, lr_lambda=self.lr_decay_lambda
        )

        #
        self.to(self._dtype).to(self.device)



    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        a, metaData = self.actor(state)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch: dict, progress: float):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        self.progress = progress

        states = self.to_tensor(batch["states"])
        controls = self.to_tensor(batch["controls"])
        rewards = self.to_tensor(batch["rewards"])
        terminations = self.to_tensor(batch["terminations"])
        truncations = self.to_tensor(batch["truncations"])
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
                truncations=truncations,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        l2_losses = []
        entropy_losses = []

        clip_frcontrols = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_controls = states[indices], controls[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]

                # 1. Critic Loss (with optional regularization)
                value_loss, l2_loss = self.critic_loss(mb_states, mb_returns)
                # Track value loss for logging
                value_losses.append(value_loss.item())
                l2_losses.append(l2_loss.item())

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

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss + l2_loss
                losses.append(loss.item())

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        supp_dict = {}
        loss_dict = {
            f"{self.name}/RL_loss/loss": np.mean(losses),
            f"{self.name}/RL_loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/RL_loss/value_loss": np.mean(value_losses),
            f"{self.name}/RL_loss/l2_loss": np.mean(l2_losses),
            f"{self.name}/RL_loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/lr/actor_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/RL_analytics/std_dev": self.actor.logstd.exp().mean().item(),
            f"{self.name}/RL_analytics/clip_fraction": np.mean(clip_frcontrols),
            f"{self.name}/RL_analytics/kl_divergence": (
                sum(target_kl) / len(target_kl) if target_kl else 0.0
            ),
            f"{self.name}/RL_analytics/K-epoch": k + 1,
            f"{self.name}/RL_analytics/avg_rewards": torch.mean(rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        loss_dict.update(grad_dict)

        self.ppo_lr_scheduler.step()
        self.actor.anneal_stddev(progress, mode="exponential")

        # Cleanup
        del states, controls, rewards, terminations, old_logprobs
        self.eval()

        update_time = time.time() - t0

        return loss_dict, supp_dict, update_time

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

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self._entropy_scaler * entropy.mean()

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
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss
