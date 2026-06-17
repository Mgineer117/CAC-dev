import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

from typing import Union
from policy.base import Base
from policy.layers.building_blocks import MLP, SirenNet


class CCM_Generator(nn.Module):
    """
    CCM_Generator generates a state-dependent matrix W(x) modeled using a Gaussian distribution.
    It leverages a neural network (MLP) to learn parameters of the Gaussian distribution
    (mean and covariance) and constructs a positive semi-definite matrix W(x) by sampling
    from the distribution and ensuring symmetry.

    Attributes:
        x_dim (int): Dimension of the state space.
        device (str): Computational device ('cpu' or 'cuda').
        w_lb (float): Lower bound added to the diagonal of W to ensure positive definiteness.
        model (MLP): Multi-layer perceptron (MLP) that generates parameters for the Gaussian distribution.
        mu (torch.nn.Linear): Linear layer generating the mean of the distribution.
        logstd (torch.nn.Linear): Linear layer generating the log of the standard deviation of the distribution.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list,
        activation: Union[str, nn.Module] = nn.Tanh(),
        mode: str = "stochastic",
        device: str = "cpu",
    ):
        super().__init__()

        # Initializing model parameters
        self.x_dim = x_dim
        self.mode = mode
        self.device = device

        # Define the model with given input dimension and hidden layers
        if isinstance(activation, str) and activation.lower() == "siren":
            self.model = SirenNet(input_dim=x_dim, hidden_dims=hidden_dim, device=device)
        else:
            if isinstance(activation, str):
                if activation.lower() == "tanh":
                    activation = nn.Tanh()
                elif activation.lower() == "relu":
                    activation = nn.ReLU()
                else:
                    raise ValueError(f"Unknown activation: {activation}")
            self.model = MLP(input_dim=x_dim, hidden_dims=hidden_dim, activation=activation, device=device)

        # Linear layers for the mean (mu) and log-std (logstd) of the Gaussian distribution
        self.mu = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)
        self.logstd = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ):
        """
        Forward pass to compute the matrix W(x) based on Gaussian distribution parameters.

        The output matrix W(x) is sampled from a Gaussian distribution with mean (mu)
        and variance (given by the exp(logstd)). If `deterministic` is True, the output
        matrix is set to the mean (mu). Otherwise, a random sample is drawn.

        Args:
            states (torch.Tensor): Input tensor representing the current state(s).
            deterministic (bool): If True, the output is the mean; otherwise, it is a sample.

        Returns:
            W (torch.Tensor): Computed matrix W(x) of shape (n, x_dim, x_dim).
            dict (dict): A dictionary containing distribution information (log probabilities, entropy, etc.)
        """
        n = x.shape[0]

        # Generate logits from the input states via the MLP
        logits = self.model(x)
        # Calculate mean (mu) and log standard deviation (logstd)
        mu = self.mu(logits)

        # If deterministic, use the mean for W(x) and calculate corresponding log probabilities
        if self.mode == "deterministic" and deterministic:
            W = mu  # Use mean (mu) as W(x) in deterministic case
            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        else:
            logstd = self.logstd(logits)

            # Clamping logstd for numerical stability and to prevent extreme values
            logstd = torch.clamp(logstd, min=-5, max=2)
            # Calculate variance as exp(logstd)^2
            std = torch.exp(logstd)

            # change it to multivariate Gaussian
            dist = Normal(loc=mu, scale=std)

            # Sample W(x) from the distribution
            W = dist.rsample()  # Sample from the distribution

            # Calculate log-probability, probability, and entropy
            logprobs = dist.log_prob(W).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().unsqueeze(-1).sum(1)

        # Reshape W(x) to the desired shape (n, x_dim, x_dim) and ensure symmetry
        W = W.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)  # W = WᵀW ensures symmetry

        return W, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }
