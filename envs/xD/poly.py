"""
Polynomial nonlinear 2D environment for contraction theory verification.

System (continuous-time, control-affine):
    ẋ₁ = x₂
    ẋ₂ = -x₁(1 + eps * x₁²) - x₂ + u

    (Duffing-type oscillator with cubic restoring force)

SDC factorisation:  f(x) = A(x) @ x  where
    A(x) = [[0,              1 ],
            [-(1+eps*x₁²),  -1]]

Contraction metric M(x):
  • SDRE: solve CARE(A(x), B, Q, R) pointwise  (local, implemented here)
  • SOS:  solve SDP to find polynomial M(x) globally  (plug cvxpy here)

Control policies available:
  • K_sdlqr(x) = -R⁻¹ Bᵀ M(x)       (SDLQR — SD-LQR at each x)
  • RL policy from the existing training pipeline (task="poly")
"""

import numpy as np
import scipy.linalg
import torch

from envs.env_base import BaseEnv

# ── System parameters ─────────────────────────────────────────────────────────
EPS     = 0.2       # cubic coefficient (must be > 0 for global stability)
N_DIM_X = 2
N_DIM_U = 1
DT      = 0.05

B_C = np.array([[0.0], [1.0]])   # constant input matrix

# ── SDRE weights ──────────────────────────────────────────────────────────────
Q_SDRE = np.eye(N_DIM_X)
R_SDRE = 0.1 * np.eye(N_DIM_U)

# ── State / control bounds ────────────────────────────────────────────────────
X_MIN        = np.array([-5., -5.]).reshape(-1, 1)
X_MAX        = np.array([ 5.,  5.]).reshape(-1, 1)
XE_INIT_MIN  = np.array([-2., -2.])
XE_INIT_MAX  = np.array([ 2.,  2.])
XE_MIN       = np.array([-2., -2.]).reshape(-1, 1)
XE_MAX       = np.array([ 2.,  2.]).reshape(-1, 1)
XREF_INIT_MIN = np.array([0., 0.])
XREF_INIT_MAX = np.array([0., 0.])
UREF_MIN = np.array([-10.]).reshape(-1, 1)
UREF_MAX = np.array([ 10.]).reshape(-1, 1)

_BASE_CONFIG = dict(
    x_min=X_MIN, x_max=X_MAX,
    xref_init_min=XREF_INIT_MIN, xref_init_max=XREF_INIT_MAX,
    xe_init_min=XE_INIT_MIN, xe_init_max=XE_INIT_MAX,
    xe_min=XE_MIN, xe_max=XE_MAX,
    uref_min=UREF_MIN, uref_max=UREF_MAX,
    num_dim_x=N_DIM_X, num_dim_control=N_DIM_U,
    pos_dimension=1,
    dt=DT, time_bound=5.0,
    q=1.0, r=0.0,
    sample_mode="Uniform", reward_mode="default",
)


# ── Metric computation ────────────────────────────────────────────────────────

def _A_sdc(x1: float) -> np.ndarray:
    """SDC matrix A(x) s.t. f(x) = A(x) @ x (scalar x₁ input)."""
    return np.array([[0.0, 1.0],
                     [-(1.0 + EPS * x1**2), -1.0]])


def compute_sdre_metric(x, Q=Q_SDRE, R=R_SDRE):
    """
    SDRE contraction metric at state x.

    Solves CARE(A(x), B, Q, R):  Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0

    By the Riccati identity:  A_cl(x)ᵀ P + P A_cl(x) = -Q - K(x)ᵀ R K(x) ≺ 0
    → M(x) = P satisfies the continuous-time contraction condition at x.

    Returns: (K, M)  where K = -R⁻¹ Bᵀ M  (SDLQR gain at x).
    """
    x1 = float(x[0]) if hasattr(x, '__len__') else float(x)
    A  = _A_sdc(x1)
    try:
        P = scipy.linalg.solve_continuous_are(A, B_C, Q, R)
        K = -np.linalg.solve(R, B_C.T @ P)
        return K, P
    except Exception:
        return np.zeros((N_DIM_U, N_DIM_X)), np.eye(N_DIM_X)


def compute_contraction_rate(x, dt=DT, Q=Q_SDRE, R=R_SDRE):
    """
    Local contraction rate  λ(x) = -max_eig(A_cl(x)ᵀ M(x) + M(x) A_cl(x)) / 2.
    Positive λ(x) confirms local contraction at x under SDLQR control.
    """
    K, P = compute_sdre_metric(x, Q, R)
    A_cl = _A_sdc(float(x[0])) + B_C @ K
    sym  = A_cl.T @ P + P @ A_cl
    return -float(np.max(np.linalg.eigvalsh(sym))) / 2.0


def compute_linearised_metric(Q=Q_SDRE, R=R_SDRE):
    """
    SDRE metric evaluated at x=0 (linearisation point).
    Valid as a constant metric near the origin.
    Returns (K_lin, M_lin) matching the LinearEnv interface.
    """
    return compute_sdre_metric(np.zeros(N_DIM_X), Q, R)


# ── Continuous-time dynamics helpers ─────────────────────────────────────────

def _f_np(x):
    """f(x): continuous-time drift, numpy, shape (2,) or (N,2)."""
    if x.ndim == 1:
        return np.array([x[1], -x[0] * (1 + EPS * x[0]**2) - x[1]])
    return np.column_stack([x[:, 1],
                            -x[:, 0] * (1 + EPS * x[:, 0]**2) - x[:, 1]])


class PolyEnv(BaseEnv):
    """
    2D polynomial nonlinear oscillator (Duffing-type).

    Key attributes set in __init__:
        self.M_lin : constant approximation metric (SDRE at x=0)
        self.K_lin : SDLQR gain at x=0

    For per-state metric: call compute_sdre_metric(x) directly.
    Swap in a global SOS polynomial M(x) via the same interface.
    """

    task = "poly"
    angle_idx = [] 

    def __init__(
        self,
        sample_mode: str = "Uniform",
        reward_mode: str = "default",
    ):
        cfg = {**_BASE_CONFIG,
               "sample_mode": sample_mode,
               "reward_mode": reward_mode}
        super().__init__(cfg)
        self.K_lin, self.M_lin = compute_linearised_metric()

    # ── dynamics ──────────────────────────────────────────────────────────────

    def _f_logic(self, x, lib):
        if lib is torch:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x1, x2 = x[:, 0:1], x[:, 1:2]
            return torch.cat([x2, -x1 * (1 + EPS * x1**2) - x2], dim=-1)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return _f_np(x)

    def _B_logic(self, x, lib):
        if lib is torch:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]
            B = torch.tensor(B_C, dtype=x.dtype, device=x.device)
            return B.unsqueeze(0).expand(n, -1, -1)
        n = x.shape[0] if x.ndim == 2 else 1
        return np.tile(B_C[np.newaxis], (n, 1, 1))

    # ── required abstract methods ─────────────────────────────────────────────

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        return np.zeros(N_DIM_U)

    def system_reset(self):
        _, _, x_0 = self.define_initial_state()
        T = int(self.time_bound / self.dt)
        xref = np.zeros((T + 1, N_DIM_X))
        uref = np.zeros((T, N_DIM_U))
        return x_0, xref, uref, T
