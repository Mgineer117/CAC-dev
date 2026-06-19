import numpy as np

try:
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False


class CVSTEM:
    """CV-STEM (Convex optimization-based Steady-state Tracking Error Minimization),
    after Tsukamoto, Chung & Slotine (astrohiro/ncm; IEEE L-CSS 2021).

    Control case. Given state-dependent coefficient matrices A(x) (we use the
    Jacobian df/dx as the SDC) and input matrices B(x) at a set of sampled states,
    CV-STEM solves a SINGLE convex SDP over all samples that share the scalars
    ``nu`` (control-authority) and ``chi`` (condition-number bound) to minimize an
    upper bound on the steady-state tracking error:

        variables : W_k (PSD, = M_k^{-1}) for each sample k,  nu >= 0,  chi >= 0
        minimize  : (1/alpha) * chi  +  w_nu * nu
        s.t.       I  <=  W_k  <=  chi I                              (forall k)
                   -2 alpha W_k
                     - ( (W_k - I)/dt + W_k A_k^T + A_k W_k - 2 nu B_k B_k^T )
                     >>  eps I                                        (forall k)

    The contraction rate ``alpha`` is selected by a line search that increases
    alpha until the objective stops improving (CV-STEM ``linesearch``). The optimal
    metrics are M_k = W_k^{-1}; the NCM then regresses these (their Cholesky
    factors) with a neural network. The realized controller is u = u* - B^T M e.

    Notes vs. the reference: MOSEK (used in the paper) requires a license, so we
    solve with CLARABEL/SCS; set ``include_dwdt=False`` for the steady-state
    (constant-metric) variant, which is far better conditioned when the simulation
    dt is tiny.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        dt: float,
        alpha: float = 0.5,
        w_nu: float = 1.0,
        epsilon: float = 1e-4,
        linesearch: bool = True,
        alpha_min: float = 0.05,
        alpha_max: float = 2.0,
        alpha_step: float = 0.1,
        include_dwdt: bool = True,
        device: str = "cpu",
    ):
        if not _HAS_CVXPY:
            raise ImportError(
                "CV-STEM requires cvxpy. Install it with `pip install cvxpy`."
            )
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt
        self.alpha = alpha
        self.w_nu = w_nu
        self.epsilon = epsilon
        self.linesearch = linesearch
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.include_dwdt = include_dwdt

    def _solve_sdp(self, A: np.ndarray, B: np.ndarray, alpha: float):
        """Solve the joint CV-STEM SDP for a fixed contraction rate alpha.

        Args:
            A: (K, n, n) state Jacobians df/dx (SDC).
            B: (K, n, m) input matrices.
        Returns:
            (Ws, nu, chi, J, status) with Ws: (K, n, n) or None if infeasible.
        """
        n, m = self.x_dim, self.u_dim
        K = A.shape[0]
        I = np.eye(n)
        eps = self.epsilon

        Ws = [cp.Variable((n, n), PSD=True) for _ in range(K)]
        nu = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)

        constraints = []
        for k in range(K):
            Ak, Bk, W = A[k], B[k], Ws[k]
            constraints += [chi * I - W >> 0, W - I >> 0]
            dwdt = ((W - I) / self.dt) if self.include_dwdt else 0
            constraints += [
                -2 * alpha * W
                - (dwdt + W @ Ak.T + Ak @ W - 2 * nu * (Bk @ Bk.T))
                >> eps * I
            ]

        J = chi / alpha + self.w_nu * nu
        prob = cp.Problem(cp.Minimize(J), constraints)

        status = self._robust_solve(prob)
        if Ws[0].value is None:
            return None, None, None, np.inf, status

        Ws_val = np.stack([W.value for W in Ws], axis=0)
        return Ws_val, float(nu.value), float(chi.value), float(prob.value), status

    @staticmethod
    def _robust_solve(prob):
        """Try CLARABEL then SCS (MOSEK needs a license)."""
        for solver in (cp.CLARABEL, cp.SCS):
            try:
                prob.solve(solver=solver)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return prob.status
            except Exception:
                continue
        return prob.status if prob.status is not None else "failed"

    def solve(self, A: np.ndarray, B: np.ndarray):
        """Solve CV-STEM, line-searching alpha to minimize the steady-state bound.

        Returns:
            Ws: (K, n, n) optimal dual metrics.
            info: dict with alpha, nu, chi, J, status.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        if not self.linesearch:
            Ws, nu, chi, J, status = self._solve_sdp(A, B, self.alpha)
            return Ws, {
                "alpha": self.alpha,
                "nu": nu,
                "chi": chi,
                "J": J,
                "status": status,
            }

        best = None
        alphas = np.arange(self.alpha_min, self.alpha_max + 1e-9, self.alpha_step)
        prev_J = np.inf
        for alpha in alphas:
            Ws, nu, chi, J, status = self._solve_sdp(A, B, float(alpha))
            if Ws is not None and J < (best["J"] if best else np.inf):
                best = {
                    "Ws": Ws,
                    "alpha": float(alpha),
                    "nu": nu,
                    "chi": chi,
                    "J": J,
                    "status": status,
                }
            # CV-STEM line search: stop once the objective starts increasing.
            if Ws is not None and J > prev_J and best is not None:
                break
            if Ws is not None:
                prev_J = J

        if best is None:
            raise RuntimeError(
                "CV-STEM SDP infeasible for all alpha in the line search. "
                "Try include_dwdt=False or a larger cvstem_dt."
            )

        Ws = best.pop("Ws")
        return Ws, best
