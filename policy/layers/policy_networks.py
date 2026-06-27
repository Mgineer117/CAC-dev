from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from policy.layers.building_blocks import MLP, SirenNet


def get_activation(activation):
    """Resolves an activation given either an nn.Module or a string name."""
    if isinstance(activation, nn.Module):
        return activation
    name = str(activation).lower()
    table = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),
    }
    if name not in table:
        raise ValueError(f"Unknown actor activation: {activation}")
    return table[name]


def get_u_model(
    x_dim: int,
    u_dim: int,
    hidden_dims: list = None,
    activation=nn.Tanh(),
):
    """
    Constructs two neural networks (w1 and w2) that generate dynamic weight matrices
    based on the trimmed current and reference states. These networks are task-agnostic
    and used to compute control-relevant transformations in the C3M_U controller.

    Args:
        x_dim (int): Full state dimension.
        u_dim (int): Dimension of the action space.
        hidden_dims (list): Hidden layer widths of the weight-generator MLPs.
        activation: Activation (nn.Module or name) used inside the generators.

    Returns:
        w1 (nn.Module): Network mapping input to a flattened tensor of shape (c * x_dim),
                        later reshaped to (c, x_dim) for transforming the error vector.
        w2 (nn.Module): Network mapping input to a flattened tensor of shape (c * u_dim),
                        later reshaped to (u_dim, c) to map the transformed error to control.
    """
    if hidden_dims is None:
        hidden_dims = [128]

    input_dim = 2 * x_dim  # Concatenated trimmed x and x_ref
    c = 3 * x_dim  # Intermediate dimension multiplier

    # SIREN backbone (sinusoidal activations) is a network type rather than a
    # pointwise activation, so it is built directly instead of via get_activation.
    if isinstance(activation, str) and activation.lower() == "siren":
        w1 = SirenNet(input_dim, list(hidden_dims), c * x_dim)
        w2 = SirenNet(input_dim, list(hidden_dims), c * u_dim)
        return w1, w2

    activation = get_activation(activation)

    # First weight generator (for projecting error vector to latent space)
    w1 = MLP(input_dim, list(hidden_dims), c * x_dim, activation=activation)

    # Second weight generator (for projecting latent to action space)
    w2 = MLP(input_dim, list(hidden_dims), c * u_dim, activation=activation)

    return w1, w2


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def trim_state(self, state: torch.Tensor):
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor):
        pass

    def anneal_stddev(self, progress: float, mode: str = "exponential"):
        if self.anneal:
            # Target value (approx sigma = 2e-9)
            final_logstd = torch.tensor(-20.0)

            # Ensure progress is clamped 0-1
            progress = min(max(progress, 0.0), 1.0)

            with torch.no_grad():
                if mode == "linear":
                    # Linear Interpolation
                    new_logstd = (
                        self.init_logstd * (1.0 - progress) + final_logstd * progress
                    )

                elif mode == "exponential":
                    # Curve shape: Stays near init_logstd longer, then drops fast at the end.
                    exponent = 5.0
                    ratio = progress**exponent

                    new_logstd = self.init_logstd * (1.0 - ratio) + final_logstd * ratio

                else:
                    raise ValueError(f"Unknown annealing mode: {mode}")

                # Clip for safety (Standard PPO bounds)
                new_logstd = torch.clamp(new_logstd, -20, 2)
                self.logstd.data.copy_(new_logstd)

    def log_prob(self, dist: torch.distributions, controls: torch.Tensor):
        """
        Computes log probability of given controls under the distribution.

        Args:
            dist (torch.distributions): The distribution of controls.
            controls (torch.Tensor): The controls for which to compute the log probability.

        Returns:
            logprobs (torch.Tensor): The log probability of the controls.
        """
        if self.mode == "stochastic":
            controls = controls.squeeze() if controls.shape[-1] > 1 else controls
            logprobs = dist.log_prob(controls).unsqueeze(-1).sum(1)
            return logprobs
        else:
            raise ValueError(
                f"Log probability computation is only valid in 'stochastic' mode, not '{self.mode}'"
            )

    def entropy(self, dist: torch.distributions):
        """
        For code consistency, computes entropy of the distribution.

        Args:
            dist (torch.distributions): The distribution to compute entropy for.

        Returns:
            entropy (torch.Tensor): The entropy of the distribution.
        """
        if self.mode == "stochastic":
            return dist.entropy().unsqueeze(-1).sum(1)
        else:
            raise ValueError(
                f"Entropy computation is only valid in 'stochastic' mode, not '{self.mode}'"
            )


class CLActor(BaseActor):
    """
    C3M_U: Control model to predict control input 'u' based on state, reference state,
    and learned task-specific parameters using neural networks.

    The model generates weight matrices from the trimmed states via neural networks,
    and applies them to the error between current and reference states to compute the control action.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        mode: str = "deterministic",
        anneal_stddev: bool = False,
        hidden_dim: list = None,
        activation=nn.Tanh(),
    ):
        """
        Initialize the control model.

        Args:
            x_dim (int): Dimension of the state vector x.
            state_dim (int): Total dimension of the combined state vector.
            u_dim (int): Dimension of the control/action vector u.
            mode (str): Mode of operation, either "deterministic" or "stochastic".
            hidden_dim (list): Hidden widths of the weight-generator MLPs.
            activation: Activation (nn.Module or name) for the generators.
        """
        super().__init__()

        self.x_dim = x_dim  # Dimension of state x
        self.u_dim = u_dim  # Dimension of action u

        self.mode = mode
        assert mode in [
            "deterministic",
            "stochastic",
        ], "Mode must be 'deterministic' or 'stochastic'"

        # Obtain task-specific neural networks that generate weight matrices
        self.w1, self.w2 = get_u_model(
            x_dim, u_dim, hidden_dims=hidden_dim, activation=activation
        )
        self.init_logstd = torch.zeros(1, u_dim)

        #
        self.anneal = anneal_stddev
        self.logstd = nn.Parameter(
            self.init_logstd.clone(), requires_grad=not self.anneal
        )

    def trim_state(self, state: torch.Tensor):
        """Trims a state tensor into its components (x, xref, uref)."""
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.u_dim]

        return x, xref, uref

    def forward(self, state: torch.Tensor):
        """
        Forward pass to compute control input u.

        Args:
            x (torch.Tensor): Current state x, shape (batch_size, x_dim)
            xref (torch.Tensor): Reference state x_ref, shape (batch_size, x_dim)
            uref (torch.Tensor): Reference control input u_ref, unused here
            deterministic (bool): Placeholder for compatibility; unused

        Returns:
            u (torch.Tensor): Computed control input, shape (batch_size, u_dim)
            dict: Empty dictionary (placeholder for potential future use)
        """
        x, xref, uref = self.trim_state(state)
        x_xref = torch.cat((x, xref), axis=-1)
        n = x.shape[0]  # Batch size

        # Compute the error between x and x_ref
        e = (x - xref).unsqueeze(-1)  # Shape: (batch_size, x_dim, 1)

        # Generate weight matrices from the neural networks
        w1 = self.w1(x_xref).reshape(
            n, -1, self.x_dim
        )  # Shape: (batch_size, x_dim, x_dim)
        w2 = self.w2(x_xref).reshape(
            n, self.u_dim, -1
        )  # Shape: (batch_size, u_dim, x_dim)

        # Compute intermediate representation
        l1 = F.tanh(torch.matmul(w1, e))  # Shape: (batch_size, hidden_dim, 1)
        mu = torch.matmul(w2, l1).squeeze(-1)  # Shape: (batch_size, u_dim)

        if self.mode == "deterministic":
            # For deterministic controls, return the mean of the distribution
            u = mu

            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        elif self.mode == "stochastic":
            logstd = torch.clip(
                self.logstd, -20, 2
            )  # Clip logstd to avoid numerical issues
            std = torch.exp(logstd.expand_as(mu))
            dist = Normal(loc=mu, scale=std)

            u = dist.rsample()

            logprobs = dist.log_prob(u).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().sum(1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return u, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }


class RLActor(BaseActor):

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: list,
        mode: str = "deterministic",
        anneal_stddev: bool = False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.x_dim, self.u_dim = x_dim, u_dim
        input_dim = 2 * x_dim + u_dim  # Concatenated x and x_ref
        # Initialize the model: MLP that outputs controls
        self.model = MLP(
            input_dim,
            hidden_dim,
            u_dim,
            activation=get_activation(activation),
            initialization="actor",
        )
        self.init_logstd = torch.zeros(1, u_dim)

        #
        self.anneal = anneal_stddev
        self.logstd = nn.Parameter(
            self.init_logstd.clone(), requires_grad=not self.anneal
        )

        self.mode = mode
        assert mode in [
            "deterministic",
            "stochastic",
        ], "Mode must be 'deterministic' or 'stochastic'"

    def trim_state(self, state: torch.Tensor):
        """Trims a state tensor into its components (x, xref, uref)."""
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim :]

        return x, xref, uref

    def forward(self, state: torch.Tensor):
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits

        if self.mode == "deterministic":
            # For deterministic controls, return the mean of the distribution
            dist = None
            a = mu
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        elif self.mode == "stochastic":
            logstd = torch.clip(
                self.logstd, -20, 2
            )  # Clip logstd to avoid numerical issues
            std = torch.exp(logstd.expand_as(mu))
            dist = Normal(loc=mu, scale=std)
            a = dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().sum(1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }


class RLCritic(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: list, activation="tanh"):
        super().__init__()

        # Initialize the model: MLP that outputs the value function (1 output)
        self.model = MLP(
            input_dim, hidden_dim, 1, activation=get_activation(activation), initialization="critic"
        )

    def forward(self, state: torch.Tensor):
        # Pass the state through the model to get the value
        value = self.model(state)
        return value
