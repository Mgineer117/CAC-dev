"""Network layers for Soft Actor-Critic (SAC).

Two pieces, mirroring the rest of ``policy/layers``:

  * :class:`SACActor` — a tanh-squashed diagonal-Gaussian policy with a
    state-dependent log-std head.  The pre-tanh Gaussian is reparameterised
    (rsample) so the actor loss is pathwise-differentiable, and the tanh
    Jacobian correction is applied to the log-probability.  Actions are scaled
    into the environment's residual-control range via fixed ``action_scale`` /
    ``action_bias`` buffers.
  * :class:`QCritic` — a state-action value network Q(s, a) built on the shared
    :class:`MLP` backbone.

The actor exposes the same ``forward(state) -> (a, metaData)`` contract as the
on-policy actors (so the multiprocessing sampler can drive it unchanged) plus
SAC-specific helpers: :meth:`sample` (action + log-prob for the SAC update) and
:meth:`mean_control` (the deterministic squashed mean, used to form the
differential feedback gain K = du/dx in the contraction loss).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from policy.layers.building_blocks import MLP
from policy.layers.policy_networks import BaseActor, get_activation

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SACActor(BaseActor):
    """Tanh-squashed Gaussian policy for SAC.

    Args:
        x_dim:        current-state dimension (used by :meth:`trim_state`).
        u_dim:        action / control dimension.
        state_dim:    full observation dimension fed to the network.
        hidden_dim:   MLP hidden widths.
        action_scale: per-dim half-range of the (residual) action box; the tanh
                      output is multiplied by this. Scalar or array of len u_dim.
        action_bias:  per-dim centre of the action box.
        activation:   activation name or module for the MLP backbone.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        state_dim: int,
        hidden_dim: list,
        action_scale=1.0,
        action_bias=0.0,
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.state_dim = state_dim
        # SAC samples a squashed Gaussian; "mode" is kept for interface parity but
        # the actor is always stochastic at collection time.
        self.mode = "stochastic"
        self.anneal = False

        self.backbone = MLP(
            state_dim,
            list(hidden_dim),
            activation=get_activation(activation),
            initialization="actor",
        )
        h = self.backbone.output_dim
        self.mu_head = nn.Linear(h, u_dim)
        self.logstd_head = nn.Linear(h, u_dim)

        action_scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)
        action_bias = np.asarray(action_bias, dtype=np.float32).reshape(-1)
        if action_scale.size == 1:
            action_scale = np.full(u_dim, action_scale.item(), dtype=np.float32)
        if action_bias.size == 1:
            action_bias = np.full(u_dim, action_bias.item(), dtype=np.float32)
        self.register_buffer("action_scale", torch.from_numpy(action_scale))
        self.register_buffer("action_bias", torch.from_numpy(action_bias))

    def trim_state(self, state: torch.Tensor):
        """Splits a state into (x, xref, uref) for the tracking reward."""
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim :]
        return x, xref, uref

    def _mu_logstd(self, state: torch.Tensor):
        feat = self.backbone(state)
        mu = self.mu_head(feat)
        logstd = torch.clamp(self.logstd_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mu, logstd

    def _squash(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh) * self.action_scale + self.action_bias

    def sample(self, state: torch.Tensor):
        """Reparameterised action + summed log-prob with the tanh correction.

        Returns (action, logprob) with action shape (n, u_dim), logprob (n, 1).
        """
        mu, logstd = self._mu_logstd(state)
        std = logstd.exp()
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        tanh = torch.tanh(pre_tanh)
        action = tanh * self.action_scale + self.action_bias

        # log_prob with change-of-variables for tanh; the numerically stable form
        # log(1 - tanh^2) = 2*(log2 - x - softplus(-2x)).
        logprob = dist.log_prob(pre_tanh)
        logprob -= 2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        logprob -= torch.log(self.action_scale)
        logprob = logprob.sum(-1, keepdim=True)
        return action, logprob

    def mean_control(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic squashed mean action (differentiable in ``state``).

        Used both for evaluation/deployment and to form K = du/dx inside the
        contraction loss, so it deliberately keeps the autograd graph.
        """
        mu, _ = self._mu_logstd(state)
        return self._squash(mu)

    def forward(self, state: torch.Tensor):
        """Stochastic during training, deterministic mean during eval."""
        if not self.training:
            action = self.mean_control(state)
            zeros = torch.zeros(action.shape[0], 1, device=state.device)
            return action, {"dist": None, "probs": zeros, "logprobs": zeros, "entropy": zeros}
        action, logprob = self.sample(state)
        return action, {
            "dist": None,
            "probs": torch.exp(logprob),
            "logprobs": logprob,
            "entropy": -logprob,
        }


class QCritic(nn.Module):
    """State-action value network Q(s, a)."""

    def __init__(self, state_dim: int, u_dim: int, hidden_dim: list):
        super().__init__()
        self.model = MLP(
            state_dim + u_dim,
            list(hidden_dim),
            1,
            activation=nn.ReLU(),
            initialization="critic",
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([state, action], dim=-1))
