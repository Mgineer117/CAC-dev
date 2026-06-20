"""
Linear time-invariant environment for contraction theory verification.

System (damped harmonic oscillator, 2-state):
    ẋ = A_c x + B_c u
    A_c = [[0,       1     ],
           [-ω²,  -2ζω  ]]
    B_c = [[0], [1]]

Reference: constant zero equilibrium (x_ref = 0, u_ref = 0).
Exact contraction metric M = Riccati solution P from LQR(Q, R).
"""

import numpy as np
import scipy.linalg
import torch

from envs.env_base import BaseEnv

# ── System parameters ─────────────────────────────────────────────────────────
OMEGA = 2.0    # natural frequency  (rad/s)
ZETA  = 0.5    # damping ratio

A_C = np.array([[0.0,          1.0         ],
                [-OMEGA**2,  -2*ZETA*OMEGA ]])
B_C = np.array([[0.0], [1.0]])

N_DIM_X = 2
N_DIM_U = 1
DT      = 0.05

# ── LQR weights for the exact metric ─────────────────────────────────────────
Q_LQR = np.eye(N_DIM_X)
R_LQR = 0.1 * np.eye(N_DIM_U)

# ── State / control bounds ────────────────────────────────────────────────────
X_MIN        = np.array([-5., -5.]).reshape(-1, 1)
X_MAX        = np.array([ 5.,  5.]).reshape(-1, 1)
XE_INIT_MIN  = np.array([-2., -2.])
XE_INIT_MAX  = np.array([ 2.,  2.])
XE_MIN       = np.array([-2., -2.]).reshape(-1, 1)
XE_MAX       = np.array([ 2.,  2.]).reshape(-1, 1)
XREF_INIT_MIN = np.array([0., 0.])
XREF_INIT_MAX = np.array([0., 0.])   # zero equilibrium
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
    num_windows=1,
)


def _zoh(A_c, B_c, dt):
    """Zero-order-hold exact discretization."""
    n, m = A_c.shape[0], B_c.shape[1]
    block = np.block([[A_c, B_c], [np.zeros((m, n + m))]])
    M = scipy.linalg.expm(block * dt)
    return M[:n, :n], M[:n, n:]


def compute_exact_metric(
    A_c=A_C, B_c=B_C, Q=Q_LQR, R=R_LQR, dt=DT
):
    """
    Returns (K, M, lam, A_cl, A_disc, B_disc) where:
      K      : LQR discrete gain  (u = K x)
      M      : exact contraction metric (= Riccati P)
      lam    : contraction rate λ under K
      A_cl   : closed-loop discrete matrix A + B K
      A_disc : open-loop discrete A
      B_disc : discrete B
    """
    A, B = _zoh(A_c, B_c, dt)
    P  = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K  = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    A_cl = A + B @ K

    # Contraction rate: max generalized eigenvalue of (A_cl^T M A_cl, M)
    eig  = scipy.linalg.eigvalsh(A_cl.T @ P @ A_cl, P)
    rho  = float(np.max(eig))
    lam  = -np.log(max(rho, 1e-14)) / (2.0 * dt)
    return K, P, lam, A_cl, A, B


class LinearEnv(BaseEnv):
    """
    2-state damped harmonic oscillator with constant zero reference.
    Exact contraction metric accessible via self.M; LQR gain via self.K.
    """

    task = "linear"
    angle_idx = []

    def __init__(
        self,
        sample_mode: str = "Uniform",
        reward_mode: str = "default",
        num_windows: int = 1,
    ):
        cfg = {**_BASE_CONFIG,
               "sample_mode": sample_mode,
               "reward_mode": reward_mode,
               "num_windows": num_windows}
        super().__init__(cfg)

        self.K, self.M, self.lam, self.A_cl, self.A_disc, self.B_disc = (
            compute_exact_metric()
        )

    # ── dynamics ──────────────────────────────────────────────────────────────

    def _f_logic(self, x, lib):
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib is torch else x[np.newaxis, :]
        if lib is torch:
            A = torch.tensor(A_C, dtype=x.dtype, device=x.device)
            return (A @ x.unsqueeze(-1)).squeeze(-1)
        return (A_C @ x.T).T

    def _B_logic(self, x, lib):
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib is torch else x[np.newaxis, :]
        n = x.shape[0]
        if lib is torch:
            B = torch.tensor(B_C, dtype=x.dtype, device=x.device)
            return B.unsqueeze(0).expand(n, -1, -1)
        return np.tile(B_C[np.newaxis], (n, 1, 1))

    # ── required abstract methods ─────────────────────────────────────────────

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        return np.zeros(N_DIM_U)

    def system_reset(self):
        _, xe_0, x_0 = self.define_initial_state()
        T = int(self.time_bound / self.dt)
        xref = np.zeros((T + 1, N_DIM_X))
        uref = np.zeros((T, N_DIM_U))
        return x_0, xref, uref, T
