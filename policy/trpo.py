import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from utils.functions import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        tot_count = self.count + count

        new_mean = self.mean + delta * count / tot_count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + np.square(delta) * self.count * count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class TRPO(Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        latent_dim: int,
        actor: nn.Module,
        critic: nn.Module,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        l2_reg: float = 1e-5,
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super().__init__()

        # constants
        self.name = "Algorithm"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = actor.u_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.target_kl = target_kl

        self.nupdates = nupdates

        # trainable networks
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.progress = 0.0
        # Critic LR follows the shared linear-decay schedule (the actor is updated
        # by TRPO's natural-gradient line search, not this optimizer).
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_decay_lambda)
        self.cg_steps = 15
        self.cg_damping = damping
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters

        self.RunningMeanStd = RunningMeanStd(shape=(1,))

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
        """Performs a single training step using TRPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        self.progress = progress

        states = self.to_tensor(batch["states"])
        controls = self.to_tensor(batch["controls"])
        rewards = self.to_tensor(batch["rewards"])
        terminations = self.to_tensor(batch["terminations"])
        truncations = self.to_tensor(batch["truncations"])
        old_logprobs = self.to_tensor(batch["logprobs"])

        self.RunningMeanStd.update(rewards.clone().cpu())
        std = torch.sqrt(torch.from_numpy(self.RunningMeanStd.var).to(self.device))
        rewards = rewards / (std + 1e-6)

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

        # Critic Update
        epochs = 5
        batch_size, N = states.shape[0], 5
        minibatch_size = batch_size // self.num_minibatch
        avg_value_loss = 0.0
        for _ in range(epochs):
            for _ in range(N):
                indices = np.random.choice(
                    batch_size, size=minibatch_size, replace=False
                )

                value_loss, l2_loss = self.critic_loss(
                    states[indices], returns[indices]
                )
                loss = value_loss + l2_loss
                avg_value_loss += value_loss.item() / (N * epochs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        old_actor = deepcopy(self.actor)

        def get_actor_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                _, infos = self.actor(states)
                logprobs = self.actor.log_prob(infos["dist"], controls)
                # Standard CPO objective: importance sampling ratio * advantage
                ratio = torch.exp(logprobs - old_logprobs)
                action_loss = -advantages * ratio
                return action_loss.mean()

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, states)

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.cg_damping, v)

        # === PERFORM CPO UPDATE === #
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        # 1. Compute Gradients and Step Directions
        # Policy Gradient (g)
        loss = get_actor_loss()
        grads = torch.autograd.grad(loss, actor_params)
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # Search Direction (H^-1 * g)
        stepdir, cg_error, consistency = conjugate_gradients(
            Hv, -loss_grad, self.cg_steps
        )

        # Compute the step scale (Lagrange multiplier approximation)
        # stepdir is already H^-1 * (-loss_grad) from conjugate_gradients
        shs = 0.5 * stepdir.dot(Hv(stepdir))
        lm = torch.sqrt(shs / self.target_kl)

        # DEFINE TRPO STEP
        opt_stepdir = stepdir / (lm + 1e-8)

        # 2. Line Search to satisfy KL constraint
        with torch.no_grad():
            old_params = flat_params(self.actor)
            expected_improve = -loss_grad.dot(opt_stepdir)

            # Line search inner loop
            fval = get_actor_loss(volatile=True).item()
            for i in range(self.backtrack_iters):
                fraction = self.backtrack_coeff**i
                new_params = old_params + fraction * opt_stepdir
                set_flat_params(self.actor, new_params)

                # Re-evaluate objectives
                fval_new = get_actor_loss(volatile=True).item()

                actual_improve = fval - fval_new
                current_expected_improve = expected_improve * fraction
                ratio = actual_improve / (current_expected_improve + 1e-8)
                kl = compute_kl(old_actor, self.actor, states)

                if (
                    kl <= self.target_kl
                    and ratio > 0.1
                    # and approx_new_cost <= self.cost_limit
                ):
                    break
            else:
                set_flat_params(self.actor, old_params)

        # Logging
        supp_dict = {}
        loss_dict = {
            f"{self.name}/RL_loss/value_loss": avg_value_loss,
            f"{self.name}/RL_analytics/std_dev": self.actor.logstd.exp().mean().item(),
            f"{self.name}/RL_analytics/kl_divergence": kl.item(),
            f"{self.name}/RL_analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/RL_analytics/line_search_fraction": fraction,
            f"{self.name}/RL_analytics/expected_improve": expected_improve.item(),
            f"{self.name}/RL_analytics/actual_improve": actual_improve,
            f"{self.name}/RL_analytics/reward_std": self.RunningMeanStd.var[0].item()
            ** 0.5,
        }

        # Cleanup
        del states, controls, rewards, terminations, old_logprobs
        self.eval()

        # Decay the critic LR per the shared schedule (progress set above).
        self.lr_scheduler.step()
        self.actor.anneal_stddev(progress, mode="exponential")

        update_time = time.time() - t0

        return loss_dict, supp_dict, update_time

    def critic_loss(self, mb_states: torch.Tensor, mb_returns: torch.Tensor):
        mb_values = self.critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss
