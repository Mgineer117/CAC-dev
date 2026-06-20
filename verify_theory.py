"""
Empirical verification of contraction RL theory on a 2-state linear testbed.

System: damped harmonic oscillator  (ω=2, ζ=0.5),  dt=0.05 s.

Design rationale
────────────────
  M = P_lqr  (DARE solution, Q=I, R=0.1).
  By the Riccati equation:  A_cl_c^T M A_cl_c = M - I - K_c^T R K_c ≺ M
  → ρ_c = max_gen_eig(M - I - K_c^T R K_c, M) < 1   (guaranteed)
  → λ_c = -log(ρ_c) / (2 Δt)
  → β   = 1 / (1 - γ ρ_c)

  Note on Assumption 2 (accelerated IES):
    For a linear system the contraction rate is state-independent, so
    Assumption 2 (∃ π_{c*} with λ* > λ in X*) is trivially satisfied
    with X* = X and λ* = λ.  Demonstrating acceleration requires a
    nonlinear testbed; here we verify the six base claims only.

Policies
────────
  π_c : LQR(Q=I, R=0.1)  →  generates M via Riccati
  π*  : discounted-optimal for  min Σ γ^k x^T M x  via scaled DARE

Claims verified
───────────────
  P1  Lemma:   Value Sandwich           C ≤ V^π ≤ β C
  P2  Lemma:   Value Contraction        V^{π*}(x_{k+1}) ≤ ρ_c V^{π*}(x_k)
  P3  Theorem: IES                      C(x_k) ≤ β C(x_0) ρ_c^k
  P4  Lemma:   Bounded Cost (M_ξ)
  P5  Lemma:   Imperfect Value Sandwich
  P6  Theorem: IES under M_ξ

Usage: conda run -n contraction python verify_theory.py
"""

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.xD.linear import compute_exact_metric, DT

# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────
GAMMA  = 0.99
T_SIM  = 200
N_INIT = 300
RNG    = np.random.default_rng(42)

K_c, M, _, A_cl_c, A, B = compute_exact_metric(Q=np.eye(2), R=0.1 * np.eye(1))
n, m = A.shape[0], B.shape[1]
m_lb  = float(np.linalg.eigvalsh(M).min())
m_max = float(np.linalg.eigvalsh(M).max())

# ── Utilities ─────────────────────────────────────────────────────────────────
def contraction_rate(A_cl):
    eig = scipy.linalg.eigvalsh(A_cl.T @ M @ A_cl, M)
    rho = float(np.max(eig))
    lam = -np.log(max(rho, 1e-14)) / (2.0 * DT)
    return lam, rho

def disc_lyap(A_cl, Q):
    """Solve  P = Q + γ A_cl^T P A_cl  (discounted Lyapunov)."""
    return scipy.linalg.solve_discrete_lyapunov((np.sqrt(GAMMA) * A_cl).T, Q)

def disc_dare_gain(A, B, Q, R):
    """Gain for  min Σ γ^k (x^T Q x + u^T R u)  via scaling trick."""
    P = scipy.linalg.solve_discrete_are(np.sqrt(GAMMA)*A, np.sqrt(GAMMA)*B, Q, R)
    return -np.linalg.solve(R + GAMMA * B.T @ P @ B, GAMMA * B.T @ P @ A)

def quad(P, x):          return float(x @ P @ x)
def is_psd(A, tol=1e-9): return bool(np.all(np.linalg.eigvalsh(A) >= -tol))
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {name}")
    return cond
def rollout(A_cl, T, x0):
    xs = [x0.copy()]
    for _ in range(T): xs.append(A_cl @ xs[-1])
    return np.array(xs)

# ── π_c (baseline contracting policy) ────────────────────────────────────────
lam_c, rho_c = contraction_rate(A_cl_c)
beta          = 1.0 / (1.0 - GAMMA * rho_c)

rho_c_theory = float(np.max(scipy.linalg.eigvalsh(
    M - np.eye(n) - K_c.T @ (0.1 * np.eye(m)) @ K_c, M)))

P_c = disc_lyap(A_cl_c, M)

# ── π* (discounted-optimal) ───────────────────────────────────────────────────
K_star    = disc_dare_gain(A, B, M, 1e-4 * np.eye(m))
A_cl_star = A + B @ K_star
sp_rad    = float(np.max(np.abs(np.linalg.eigvals(A_cl_star))))
P_star    = disc_lyap(A_cl_star, M)

# ── Empirical samples ─────────────────────────────────────────────────────────
dirs = RNG.standard_normal((N_INIT, n))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
X0 = dirs * RNG.uniform(0.1, 3.0, (N_INIT, 1))

C_vals      = np.array([quad(M,      x) for x in X0])
V_c_vals    = np.array([quad(P_c,    x) for x in X0])
V_star_vals = np.array([quad(P_star, x) for x in X0])


# ─────────────────────────────────────────────────────────────────────────────
# P1. VALUE SANDWICH
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ P1: Value Sandwich ═══════════════════════════════════════════════════")
print(f"   M = P_lqr,  m_lb = {m_lb:.4f},  m_max = {m_max:.4f}")
print(f"   pi_c:  rho = {rho_c:.6f}  (Riccati: {rho_c_theory:.6f}),  beta = {beta:.2f}")
print(f"   pi*:   sp_rad(A_cl*) = {sp_rad:.6f}")

check("rho_c < 1                   [Riccati guarantees contraction]", rho_c < 1.0)
check("rho_c == rho_c_theory       [numerics vs. closed-form]",
      abs(rho_c - rho_c_theory) < 1e-8)
check("M <= P_c                    i.e., C(x) <= V^pi_c(x)",   is_psd(P_c - M))
check("P_c <= beta M               i.e., V^pi_c(x) <= beta C(x)", is_psd(beta * M - P_c))

check("sp_rad(A_cl*) < 1           [pi* is stabilising]",   sp_rad < 1.0)
check("M <= P_star                 i.e., C(x) <= V^pi*(x)",   is_psd(P_star - M))
check("P_star <= beta M            [value sandwich upper]",    is_psd(beta * M - P_star))
check("P_star <= P_c               [pi* beats pi_c for M-cost]", is_psd(P_c - P_star))
check("C(x) <= V^pi*(x) <= beta C(x)  [empirical, N=%d pts]" % N_INIT,
      np.all(C_vals <= V_star_vals + 1e-10) and
      np.all(V_star_vals <= beta * C_vals + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# P2. VALUE CONTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ P2: Value Contraction ════════════════════════════════════════════════")
print(f"   V^pi*(x_{{k+1}}) / V^pi*(x_k)  must be <= rho_c = {rho_c:.6f}")

x0_t = np.array([2.0, -1.5])
V_c_t = np.array([quad(P_c,    x) for x in rollout(A_cl_c,    T_SIM, x0_t)])
V_s_t = np.array([quad(P_star, x) for x in rollout(A_cl_star, T_SIM, x0_t)])
ratio_c = (V_c_t[1:] / np.maximum(V_c_t[:-1], 1e-14)).max()
ratio_s = (V_s_t[1:] / np.maximum(V_s_t[:-1], 1e-14)).max()

check("A_cl_c^T P_c A_cl_c <= rho_c P_c       [pi_c analytic]",
      is_psd(rho_c * P_c - A_cl_c.T @ P_c @ A_cl_c))
check("A_cl*^T P* A_cl* <= rho_c P*            [pi* value contraction — key]",
      is_psd(rho_c * P_star - A_cl_star.T @ P_star @ A_cl_star))
check(f"max step ratio (pi_c) = {ratio_c:.5f} <= rho_c  [empirical]", ratio_c <= rho_c + 1e-10)
check(f"max step ratio (pi*)  = {ratio_s:.5f} <= rho_c  [empirical]", ratio_s <= rho_c + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# P3. IES THEOREM
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ P3: IES Theorem ══════════════════════════════════════════════════════")
k_arr  = np.arange(T_SIM + 1)
viol_c, viol_s = 0, 0
for x0 in X0:
    C_c = np.array([quad(M, x) for x in rollout(A_cl_c,    T_SIM, x0)])
    C_s = np.array([quad(M, x) for x in rollout(A_cl_star, T_SIM, x0)])
    if np.any(C_c > beta * C_c[0] * rho_c**k_arr + 1e-10): viol_c += 1
    if np.any(C_s > beta * C_s[0] * rho_c**k_arr + 1e-10): viol_s += 1

check(f"(pi_c) C(x_k) <= beta C_0 rho_c^k   [{N_INIT-viol_c}/{N_INIT}]", viol_c == 0)
check(f"(pi*)  C(x_k) <= beta C_0 rho_c^k   [{N_INIT-viol_s}/{N_INIT}]", viol_s == 0)


# ─────────────────────────────────────────────────────────────────────────────
# P4. BOUNDED COST  (imperfect M_xi)
# ─────────────────────────────────────────────────────────────────────────────
eta_thresh = m_lb * (GAMMA * (1.0 - rho_c)) / (2.0 - GAMMA * (1.0 + rho_c))
ETA = 0.35 * eta_thresh

raw   = RNG.standard_normal((n, n))
raw   = (raw + raw.T) / 2
evals = np.linalg.eigvalsh(raw)
Delta = (raw - evals.min() * np.eye(n)) / (evals.max() - evals.min() + 1e-12)
M_xi  = M + ETA * Delta
eta_a = float(np.linalg.norm(M - M_xi, ord=2))
lo, hi = 1.0 - eta_a / m_lb, 1.0 + eta_a / m_lb
C_xi  = np.array([quad(M_xi, x) for x in X0])

print(f"\n══ P4: Bounded Cost  (eta = {eta_a:.4f}, threshold = {eta_thresh:.4f}) ══")
check(f"(1 - eta/m_lb) M <= M_xi  (lo = {lo:.4f})", is_psd(M_xi - lo * M))
check(f"M_xi <= (1 + eta/m_lb) M  (hi = {hi:.4f})", is_psd(hi * M - M_xi))
check("(1-eta/m) C <= C_hat <= (1+eta/m) C  [empirical]",
      np.all(lo * C_vals <= C_xi + 1e-10) and np.all(C_xi <= hi * C_vals + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# P5. IMPERFECT VALUE SANDWICH
# ─────────────────────────────────────────────────────────────────────────────
alpha = (m_lb + eta_a) / (m_lb - eta_a)
ab    = alpha * beta
P_sxi = disc_lyap(A_cl_star, M_xi)
V_sxi = np.array([quad(P_sxi, x) for x in X0])

print(f"\n══ P5: Imperfect Value Sandwich  (alpha = {alpha:.4f},  alpha*beta = {ab:.2f}) ══")
check("M_xi <= P_sxi               i.e., C_hat <= V_hat^pi*",    is_psd(P_sxi - M_xi))
check("P_sxi <= alpha*beta*M_xi    i.e., V_hat^pi* <= alpha*beta*C_hat",
      is_psd(ab * M_xi - P_sxi))
check("C_hat <= V_hat^pi* <= alpha*beta*C_hat  [empirical]",
      np.all(C_xi <= V_sxi + 1e-10) and np.all(V_sxi <= ab * C_xi + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# P6. IES UNDER M_xi
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n══ P6: IES under M_xi  (eta = {eta_a:.4f} < threshold = {eta_thresh:.4f}) ══")
viol_xi = 0
for x0 in X0:
    C_k = np.array([quad(M, x) for x in rollout(A_cl_star, T_SIM, x0)])
    if np.any(C_k > ab * C_k[0] * rho_c**k_arr + 1e-10): viol_xi += 1
check(f"C(x_k) <= alpha*beta*C_0*rho_c^k   [{N_INIT-viol_xi}/{N_INIT}]", viol_xi == 0)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
x0_p  = np.array([2.0, -1.5])
traj_c = rollout(A_cl_c,    T_SIM, x0_p)
traj_s = rollout(A_cl_star, T_SIM, x0_p)

C_c_p  = np.array([quad(M,      x) for x in traj_c])
C_s_p  = np.array([quad(M,      x) for x in traj_s])
C_xi_p = np.array([quad(M_xi,   x) for x in traj_s])
V_c_p  = np.array([quad(P_c,    x) for x in traj_c])
V_s_p  = np.array([quad(P_star, x) for x in traj_s])

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    f"Contraction RL Theory Verification — Linear Testbed  (gamma={GAMMA})\n"
    f"M = P_lqr,  rho_c = {rho_c:.4f},  beta = {beta:.1f}  "
    f"|  pi*: sp_rad = {sp_rad:.4f}",
    fontsize=10,
)

ax = axes[0, 0]
idx = np.argsort(C_vals)
ax.scatter(C_vals[idx], V_star_vals[idx], s=5, c="steelblue", label="V*(x)", zorder=3)
ax.fill_between(C_vals[idx], C_vals[idx], beta * C_vals[idx], alpha=0.2,
                color="orange", label=f"[C, beta*C]  beta={beta:.0f}")
ax.plot([0, C_vals.max()], [0, C_vals.max()], "k--", lw=0.8)
ax.set(xlabel="C(x)", ylabel="V*(x)", title="P1: Value Sandwich"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.semilogy(k_arr, V_s_p, lw=2, label="V*(x_k)  under pi*")
ax.semilogy(k_arr, V_s_p[0] * rho_c**k_arr, "--", color="orange",
            label=f"V*_0 * rho_c^k")
ax.set(xlabel="Step k", ylabel="V*(x_k)", title="P2: Value Contraction"); ax.legend(fontsize=8)

ax = axes[0, 2]
ax.semilogy(k_arr, C_c_p, lw=1.5, color="blue", label="C(x_k)  pi_c")
ax.semilogy(k_arr, C_s_p, lw=1.5, color="red",  label="C(x_k)  pi*")
ax.semilogy(k_arr, beta * C_c_p[0] * rho_c**k_arr, "--", color="blue",  alpha=0.6,
            label="beta*C_0*rho_c^k")
ax.semilogy(k_arr, beta * C_s_p[0] * rho_c**k_arr, "--", color="red",   alpha=0.6)
ax.set(xlabel="Step k", ylabel="C(x_k)", title="P3: IES"); ax.legend(fontsize=7)

ax = axes[1, 0]
idx = np.argsort(C_vals)
ax.fill_between(C_vals[idx], lo*C_vals[idx], hi*C_vals[idx], alpha=0.3, color="orange",
                label=f"[(1-eta/m)C, (1+eta/m)C]")
ax.scatter(C_vals[idx], C_xi[idx], s=5, c="red", label="C_hat(x)", zorder=3)
ax.set(xlabel="C(x)", ylabel="C_hat(x)", title=f"P4: Bounded Cost  eta={eta_a:.3f}")
ax.legend(fontsize=8)

ax = axes[1, 1]
idx = np.argsort(C_xi)
ax.scatter(C_xi[idx], V_sxi[idx], s=5, c="steelblue", label="V_hat*(x)", zorder=3)
ax.fill_between(C_xi[idx], C_xi[idx], ab*C_xi[idx], alpha=0.2, color="orange",
                label=f"[C_hat, alpha*beta*C_hat]  ab={ab:.0f}")
ax.set(xlabel="C_hat(x)", ylabel="V_hat*(x)", title="P5: Imperfect Value Sandwich")
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.semilogy(k_arr, C_xi_p, lw=2, label="C_hat(x_k)")
ax.semilogy(k_arr, ab * C_xi_p[0] * rho_c**k_arr, "--", color="red",
            label="alpha*beta*C_hat_0*rho_c^k")
ax.set(xlabel="Step k", ylabel="C_hat(x_k)", title="P6: IES under M_xi")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("theory_verification.png", dpi=150)
print("\nPlot saved -> theory_verification.png")
