import math
import time
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_continuous_are
from torch import inverse, matmul, transpose
from torch.linalg import solve
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from policy.base import Base
from policy.layers.CMG_networks import CCM_Generator
from policy.layers.policy_networks import EncoderCLActor, EncoderRLCritic
from utils.functions import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)
from utils.replay_buffer import ReplayBuffer


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


class CPO(Base):
    def __init__(
        self,
        # Learning parameters
        x_dim: int,
        u_dim: int,
        dt: float,
        data: dict,
        actor: EncoderCLActor,
        critic: EncoderRLCritic,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        # CMG parameters
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        lbd: float = 1e-2,
        reward_mode: str = "default",
        # CPO parameters
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        target_kl: float = 0.03,
        # RL parameters
        num_windows: int = 1,
        gamma: float = 0.99,
        gae: float = 0.95,
        l2_reg: float = 1e-8,
        tracking_scaler: float = 1.0,
        control_scaler: float = 0.0,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super().__init__()

        self.name = "Algorithm"

        # Data and progress tracking
        self.progress, self.num_W_updates, self.num_RL_updates = 0.0, 1, 1
        self.x_dim, self.u_dim = x_dim, u_dim
        self.data = data
        self.num_minibatch, self.minibatch_size = num_minibatch, minibatch_size

        # Algorithm learning parameters
        self.lbd, self.w_ub, self.w_lb = lbd, w_ub, w_lb
        self.actor, self.reward_critic, self.cost_critic = (
            actor,
            deepcopy(critic),
            deepcopy(critic),
        )
        self.dt = dt
        self.reward_mode = reward_mode
        self.tracking_scaler = tracking_scaler
        self.control_scaler = control_scaler
        self.num_windows = num_windows
        self.gamma, self.gae, self.l2_reg = gamma, gae, l2_reg
        self.cost_limit, self.target_kl = 1000, target_kl
        self.nupdates = nupdates
        self.device = device

        # Line search parameters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.cg_damping = damping
        self.cg_steps = 15

        # Optimizers
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.reward_critic.parameters(), "lr": critic_lr},
                {"params": self.cost_critic.parameters(), "lr": critic_lr},
            ]
        )
        # Critic LR follows the shared linear-decay schedule (the actor is updated
        # by CPO's constrained natural-gradient step, not this optimizer).
        self.lr_scheduler = LambdaLR(
            self.critic_optimizer, lr_lambda=self.lr_decay_lambda
        )

        self.RunningMeanStd = RunningMeanStd(shape=(2,))

        self.to(self._dtype).to(self.device)

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

    def learn(self, batch: dict, progress: float):
        self.progress = progress

        loss_dict, supp_dict, update_time = self.learn_cpo(batch)
        self.actor.anneal_stddev(progress, mode="exponential")

        # Decay the critic LR per the shared schedule (progress set above).
        self.lr_scheduler.step()

        self.num_RL_updates += 1

        return loss_dict, supp_dict, update_time

    def learn_cpo(self, batch: dict):
        """Performs a single training step using CPO logic."""
        self.train()

        t0 = time.time()

        # === PREPARE TENSORS === #
        states = self.to_tensor(batch["states"])
        controls = self.to_tensor(batch["controls"])
        original_rewards = self.to_tensor(batch["rewards"])
        terminations = self.to_tensor(batch["terminations"])
        truncations = self.to_tensor(batch["truncations"])
        old_logprobs = self.to_tensor(batch["logprobs"])

        rewards, costs = self.get_rewards_and_costs(
            states, controls, terminations, truncations
        )

        # === COMPUTE ADVANTAGES === #
        with torch.no_grad():
            reward_values = self.reward_critic(states)
            cost_values = self.cost_critic(states)
            reward_advantages, reward_returns = estimate_advantages(
                rewards,
                terminations,
                reward_values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )
            cost_advantages, cost_returns = estimate_advantages(
                costs,
                terminations,
                cost_values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )

        # === PERFORM CRITIC UPDATE === #
        epochs = 5
        batch_size, N = states.shape[0], 5
        minibatch_size = batch_size // self.num_minibatch
        avg_reward_value_loss, avg_cost_value_loss = 0.0, 0.0
        for _ in range(epochs):
            for _ in range(N):
                indices = np.random.choice(
                    batch_size, size=minibatch_size, replace=False
                )

                reward_values = self.reward_critic(states[indices])
                cost_values = self.cost_critic(states[indices])

                reward_value_loss = self.mse_loss(
                    reward_values, reward_returns[indices]
                )
                cost_value_loss = self.mse_loss(cost_values, cost_returns[indices])

                avg_reward_value_loss += reward_value_loss.item() / (N * epochs)
                avg_cost_value_loss += cost_value_loss.item() / (N * epochs)

                value_loss = reward_value_loss + cost_value_loss

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        # === DEFINE HELPER FUNCTIONS === #
        old_actor = deepcopy(self.actor)

        def get_reward_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                _, infos = self.actor(states)
                logprobs = self.actor.log_prob(infos["dist"], controls)
                # Standard CPO objective: importance sampling ratio * advantage
                ratio = torch.exp(logprobs - old_logprobs)
                action_loss = -reward_advantages * ratio
                return action_loss.mean()

        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                _, infos = self.actor(states)
                logprobs = self.actor.log_prob(infos["dist"], controls)
                ratio = torch.exp(logprobs - old_logprobs)
                cost_loss = cost_advantages * ratio
                return cost_loss.mean()

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, states)

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.cg_damping, v)

        # Analytical definitions for lambda search (lagrangian multipliers)
        def f_a_lambda(lamda):
            a = ((r**2) / s - q) / (2 * lamda)
            b = lamda * ((cc**2) / s - self.target_kl) / 2
            c = -(r * cc) / s
            return a + b + c

        def f_b_lambda(lamda):
            a = -(q / lamda + lamda * self.target_kl) / 2
            return a

        # === PERFORM CPO UPDATE === #
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        # 1. Compute Gradients and Step Directions
        # Policy Gradient (g)
        loss = get_reward_loss()
        grads = torch.autograd.grad(loss, actor_params)
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # Cost Gradient (b)
        cost_loss = get_cost_loss()
        cost_grads = torch.autograd.grad(cost_loss, actor_params)
        cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach()

        # Search Direction (H^-1 * g)
        stepdir, cg_r_error, r_consistency = conjugate_gradients(
            Hv, -loss_grad, self.cg_steps
        )
        # Cost Search Direction (H^-1 * b)
        cost_stepdir, cg_c_error, c_consistency = conjugate_gradients(
            Hv, -cost_loss_grad, self.cg_steps
        )

        # 4. Define q, r, s for dual optimization
        q = -loss_grad.dot(stepdir)  # g^T H^-1 g
        r = loss_grad.dot(cost_stepdir)  # g^T H^-1 b
        s = -cost_loss_grad.dot(cost_stepdir)  # b^T H^-1 b

        # Calculate constraints
        current_cost = cost_returns.mean()
        limit = max(0, self.cost_limit * (1 - 1.33 * self.progress))
        cc = current_cost - limit
        # If cc > 0, we are violating.

        # Find optimal lambda and nu
        A = torch.sqrt((q - (r**2) / s) / (self.target_kl - (cc**2) / s))
        B = torch.sqrt(q / self.target_kl)

        # Check feasibility cases
        if cc > 0:  # Violation
            opt_lam_a = torch.max(r / cc, A)
            opt_lam_b = torch.max(torch.zeros_like(A), torch.min(B, r / cc))
        else:  # Safe
            opt_lam_b = torch.max(r / cc, B)
            opt_lam_a = torch.max(torch.zeros_like(A), torch.min(A, r / cc))

        opt_f_a = f_a_lambda(opt_lam_a)
        opt_f_b = f_b_lambda(opt_lam_b)

        opt_lambda = opt_lam_a if opt_f_a > opt_f_b else opt_lam_b
        opt_lambda = opt_lambda.clamp(min=1e-8)  # Avoid division by zero

        nu = (opt_lambda * cc - r) / s
        opt_nu = torch.max(nu, torch.zeros_like(nu))

        # Check for global feasibility
        if ((cc**2) / s - self.target_kl) > 0 and cc > 0:
            # Infeasible: pure cost reduction
            opt_stepdir = torch.sqrt(2 * self.target_kl / s) * cost_stepdir
        else:
            # Normal CPO update
            opt_stepdir = (stepdir - opt_nu * cost_stepdir) / opt_lambda

        # 5. Line Search
        with torch.no_grad():
            old_params = flat_params(self.actor)
            expected_improve = -loss_grad.dot(opt_stepdir)
            expected_cost_change = -cost_loss_grad.dot(opt_stepdir)

            # Line search inner loop
            fval = get_reward_loss(volatile=True).item()
            for i in range(self.backtrack_iters):
                fraction = self.backtrack_coeff**i
                new_params = old_params + fraction * opt_stepdir
                set_flat_params(self.actor, new_params)

                # Re-evaluate objectives
                fval_new = get_reward_loss(volatile=True).item()
                cost_new = get_cost_loss(volatile=True).item()  # Surrogate cost

                actual_improve = fval - fval_new
                current_expected_improve = expected_improve * fraction
                ratio = actual_improve / (current_expected_improve + 1e-8)
                approx_new_cost = current_cost + cost_new
                kl = compute_kl(old_actor, self.actor, states)

                if (
                    kl <= self.target_kl
                    and ratio > 0.1
                    # and approx_new_cost <= self.cost_limit
                ):
                    break
            else:
                set_flat_params(self.actor, old_params)

        # === LOGGING === #
        loss_dict = {
            f"{self.name}/RL_analytics/std_dev": self.actor.logstd.exp().mean().item(),
            f"{self.name}/RL_analytics/reward_std": self.RunningMeanStd.var[0].item()
            ** 0.5,
            f"{self.name}/RL_analytics/cost_std": self.RunningMeanStd.var[1].item()
            ** 0.5,
            f"{self.name}/RL_analytics/expected_improve": expected_improve.item(),
            f"{self.name}/RL_analytics/expected_cost_change": expected_cost_change.item(),
            f"{self.name}/RL_analytics/actual_improve": actual_improve,
            f"{self.name}/RL_analytics/avg_reward_value_loss": avg_reward_value_loss,
            f"{self.name}/RL_analytics/avg_cost_value_loss": avg_cost_value_loss,
            f"{self.name}/RL_analytics/kl_divergence": kl.item(),
            f"{self.name}/RL_analytics/cost_violation": cc.item(),
            f"{self.name}/RL_analytics/opt_nu": opt_nu.item(),
            f"{self.name}/RL_analytics/opt_lambda": opt_lambda.item(),
            f"{self.name}/RL_analytics/cg_r_error": cg_r_error.item(),
            f"{self.name}/RL_analytics/cg_c_error": cg_c_error.item(),
            f"{self.name}/RL_analytics/cg_r_consistency": r_consistency,
            f"{self.name}/RL_analytics/cg_c_consistency": c_consistency,
            f"{self.name}/RL_analytics/avg_rewards": torch.mean(
                original_rewards
            ).item(),
            f"{self.name}/RL_analytics/corrected_avg_rewards": torch.mean(
                rewards
            ).item(),
            f"{self.name}/RL_analytics/corrected_avg_costs": torch.mean(costs).item(),
            f"{self.name}/RL_analytics/num_backtracks": i,
            f"{self.name}/RL_analytics/cost_limit": limit,
        }
        return loss_dict, {}, time.time() - t0

    def find_exponential_curve(self, track_M_metric, terminations, truncations):
        """
        Parses batch tensors into list of trajectories based on termination/truncation masks.
        Vectorized for performance.
        """
        # 1. Identify split indices (where trajectories end)
        dones = torch.logical_or(terminations, truncations).flatten()
        # Add 1 because split happens *after* the True value
        split_indices = (torch.nonzero(dones).flatten() + 1).cpu()

        # 2. Split tensors into chunks
        # Filte out empty chunks that might occur if the last element is a termination
        track_M_traj = [
            t for t in torch.tensor_split(track_M_metric, split_indices) if len(t) > 0
        ]

        # 3. Generate exponential curves vectorized
        C = self.w_ub / self.w_lb
        exp_traj = []
        for i in range(len(track_M_traj)):
            xt = C * track_M_traj[i]

            # Create time vector t = [0, dt, 2dt, ...]
            t = torch.arange(len(xt), device=xt.device, dtype=xt.dtype) * self.dt

            # Calculate decay factor: e^(-2 * lambda * t)
            decay = torch.exp(-self.lbd * t).unsqueeze(-1)

            # Curve: x0 * decay (broadcasts across state dimensions)
            exp_curve = xt[0].unsqueeze(0) * decay
            exp_traj.append(exp_curve)

        return torch.cat(exp_traj, dim=0)

    def get_rewards_and_costs(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ):
        x, xref, uref = self.actor.trim_state(states)

        with torch.no_grad():
            I = torch.eye(self.x_dim, device=states.device)
            # Tracking Error: x - xref
            tracking_error = (x - xref).unsqueeze(-1)  # (N, x_dim, 1)
            track_M_metric = (
                tracking_error.transpose(1, 2) @ I @ tracking_error
            ).squeeze(-1)

            # Contraction error
            contraction_error = self.find_exponential_curve(
                track_M_metric, terminations, truncations
            )
            cont_M_metric = torch.relu(track_M_metric - contraction_error)

            control_reward = torch.linalg.norm(controls, dim=1, keepdim=True)

        if self.reward_mode == "default":
            tracking_reward = -self.tracking_scaler * track_M_metric
            control_reward = -self.control_scaler * control_reward
            contraction_cost = (
                cont_M_metric  # We will treat this as a cost, not a reward
            )
        elif self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + track_M_metric)
            control_reward = 1 / (1 + control_reward)
            # Since contraction_cost is a cost, we want to invert it in a way that higher costs lead to lower rewards
            contraction_cost = 1 / (1 + -cont_M_metric)

        rewards = (0.5 * tracking_reward) + (0.5 * control_reward)
        costs = contraction_cost
        self.RunningMeanStd.update(torch.cat([rewards, costs], dim=1).clone().cpu())
        std = torch.sqrt(torch.from_numpy(self.RunningMeanStd.var).to(self.device))

        rewards = rewards / (std[0] + 1e-6)
        costs = costs / (std[1] + 1e-6)

        return rewards, costs
