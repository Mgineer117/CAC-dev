"""Bounded Contraction Metric Generator.

Analogue of the action-bounding trick in RL control:

    actor:   u   = u_max * tanh( net(x) )          # unbounded logit → bounded scalar
    CMG:     W   = V @ diag( sigmoid_scaled(λ) ) @ Vᵀ   # unbounded sym matrix → bounded SPD

The network produces a flat vector that is reshaped and symmetrised into a raw
symmetric matrix S.  torch.linalg.eigh decomposes S into eigenvalues λ (which
span ℝ, so sigmoid covers the full (0,1) range) and orthogonal eigenvectors V.
A scaled sigmoid maps each eigenvalue strictly into (w_lb, w_ub):

    λ_bounded = w_lb + (w_ub - w_lb) * sigmoid(λ)

W = V diag(λ_bounded) Vᵀ is then always SPD with eigenvalues in (w_lb, w_ub)
by construction — no overshoot loss or w_lb*I addition required downstream.
"""

from typing import Union

import torch
import torch.nn as nn

from policy.layers.building_blocks import MLP, SirenNet


class BoundedCCM_Generator(nn.Module):
    """CMG with hard eigenvalue bounds baked into the forward pass.

    Replaces the WᵀW + w_lb*I + overshoot-loss pattern with a strict
    sigmoid-on-eigenvalues parameterisation, analogous to a*tanh for actions.

    Args:
        x_dim:      state dimension
        hidden_dim: list of hidden layer widths for the MLP backbone
        activation: 'tanh' | 'relu' | 'siren' | nn.Module
        mode:       'deterministic' (mean) or 'stochastic' (sampled)
        w_lb:       strict lower bound on every eigenvalue of W(x)
        w_ub:       strict upper bound on every eigenvalue of W(x)
        device:     torch device string
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list,
        activation: Union[str, nn.Module] = "tanh",
        mode: str = "deterministic",
        w_lb: float = 0.1,
        w_ub: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.x_dim = x_dim
        self.mode = mode
        self.device = device
        self.w_lb = w_lb
        self.w_ub = w_ub
        # Marker read by policies: the returned W is already SPD with eigenvalues
        # in (w_lb, w_ub), so consumers must NOT add the usual w_lb*I shift.
        self.bounded = True

        if isinstance(activation, str) and activation.lower() == "siren":
            self.model = SirenNet(input_dim=x_dim, hidden_dims=hidden_dim, device=device)
        else:
            if isinstance(activation, str):
                activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}.get(
                    activation.lower(), nn.Tanh()
                )
            self.model = MLP(
                input_dim=x_dim, hidden_dims=hidden_dim,
                activation=activation, device=device,
            )

        out_dim = x_dim * x_dim
        self.mu = nn.Linear(hidden_dim[-1], out_dim)
        self.logstd = nn.Linear(hidden_dim[-1], out_dim)

    def _to_bounded_spd(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape → symmetrise → sigmoid-on-eigenvalues → bounded SPD."""
        n = flat.shape[0]
        S_raw = flat.view(n, self.x_dim, self.x_dim)
        S = 0.5 * (S_raw + S_raw.mT)           # symmetric; eigenvalues span ℝ
        lam, V = torch.linalg.eigh(S.cpu())    # eigh unsupported on MPS; run on CPU
        lam, V = lam.to(S.device), V.to(S.device)
        lam = self.w_lb + (self.w_ub - self.w_lb) * torch.sigmoid(lam)
        return V @ torch.diag_embed(lam) @ V.mT  # SPD, λ ∈ (w_lb, w_ub)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        logits = self.model(x)
        mu = self.mu(logits)

        # Return-dict keys mirror CCM_Generator so the two are drop-in compatible.
        if self.mode == "deterministic" and deterministic:
            W = self._to_bounded_spd(mu)
            logprobs = torch.zeros(x.shape[0], 1, device=x.device)
            return W, {
                "dist": None,
                "probs": torch.ones_like(logprobs),
                "logprobs": logprobs,
                "entropy": torch.zeros_like(logprobs),
            }

        logstd = self.logstd(logits).clamp(-5, 2)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample()
        W = self._to_bounded_spd(sample)
        logprobs = dist.log_prob(sample).sum(-1, keepdim=True)
        return W, {
            "dist": dist,
            "probs": torch.exp(logprobs),
            "logprobs": logprobs,
            "entropy": dist.entropy().sum(-1, keepdim=True),
        }
