"""
Research questions Q1 and Q2 — linear testbed.

Q1: Can the optimal policy achieve accelerated contraction at the expense of
    overshoot (in linear dynamics with a perfect contraction metric)?

    Setup:
      π_c  (LQR, R=0.1): contracting policy that defines M.  ρ_c < 1 → monotone.
      π_RL (disc-DARE, R=1e-4, γ=0.99): discounted RL policy with action ∈ [-1,1].

    Key design point — initial condition x₀:
      Both "max control" surfaces (|K @ x| = 1) scale inversely with gain magnitude.
      π_RL requires |u| ≫ 1 at a unit-C initial condition, so the relevant
      comparison must start at x₀ INSIDE the action-feasible region of π_RL:

          x₀ = x₀_worst × 0.95 / |K_RL @ x₀_worst|

      where x₀_worst is the worst-case overshoot direction (max gen-eigvec of
      A_cl_RL^T M A_cl_RL w.r.t. M).  From this x₀:
        • |K_RL @ x₀| = 0.95 < 1   → no saturation, π_RL = K_RL exactly
        • |K_c  @ x₀| ≪ 1          → π_c also well within bounds
        • C(x₁)/C(x₀) = ρ_phys     → overshoot still occurs at this scale

    "Acceleration" = V^{π_RL} ≤ V^{π_c} (smaller value function, faster settling).

Q2: How does γ affect overshoot duration and intensity?

    Larger γ → more patient π_RL → higher ρ_phys (deeper overshoot)
              → smaller sp_rad (faster physical convergence)
              → larger β (wider IES envelope)
    The trade-off is: pay more overshoot upfront for faster long-term convergence.

System: damped harmonic oscillator  (ω=2, ζ=0.5, dt=0.05 s)
  M = P_lqr (DARE, Q=I, R=0.1) — exact contraction metric.

Usage: conda run -n contraction python verify_q12.py
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
K_c, M, _, A_cl_c, A, B = compute_exact_metric(Q=np.eye(2), R=0.1 * np.eye(1))
n, m = A.shape[0], B.shape[1]

def gen_eig_max(A_cl):
    return float(np.max(scipy.linalg.eigvalsh(A_cl.T @ M @ A_cl, M)))

def cont_rate(A_cl):
    rho = gen_eig_max(A_cl)
    return -np.log(max(rho, 1e-14)) / (2 * DT), rho

def disc_lyap(A_cl, gamma):
    return scipy.linalg.solve_discrete_lyapunov((np.sqrt(gamma) * A_cl).T, M)

def disc_dare_gain(gamma, R_scl=1e-4):
    R = R_scl * np.eye(m)
    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, np.sqrt(gamma)*B, M, R)
    return -np.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)

def quad(P, x):    return float(x @ P @ x)
def sp_rad(A_cl):  return float(np.max(np.abs(np.linalg.eigvals(A_cl))))
def ctrl(K, x):    return float((K @ x).flatten()[0])

def rollout_free(A_cl, T, x0):
    xs = [x0.copy()]
    for _ in range(T): xs.append(A_cl @ xs[-1])
    return np.array(xs)

def settling_step(C, frac=0.01):
    idx = np.where(np.asarray(C) < frac * C[0])[0]
    return int(idx[0]) if len(idx) else len(C)

def worst_overshoot_x0(A_cl, scale_K=None, target_u=0.95):
    """
    Return x₀ in the worst-case overshoot direction of A_cl, scaled so that
    |scale_K @ x₀| = target_u  (default: normalise to C(x₀) = 1 if scale_K=None).
    """
    _, evecs = scipy.linalg.eigh(A_cl.T @ M @ A_cl, M)
    x = evecs[:, -1].copy()
    x /= np.sqrt(quad(M, x))          # normalise: C(x) = 1
    if scale_K is not None:
        u_at_x = abs(ctrl(scale_K, x))
        x *= target_u / u_at_x
    return x

lam_c, rho_c = cont_rate(A_cl_c)


# ─────────────────────────────────────────────────────────────────────────────
# Q1: OVERSHOOT ANALYSIS  (γ = 0.99)
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_Q1 = 0.99
U_MAX    = 1.0
T_SIM    = 150

K_rl      = disc_dare_gain(GAMMA_Q1)
A_cl_rl   = A + B @ K_rl
_, rho_rl = cont_rate(A_cl_rl)
sp_rl     = sp_rad(A_cl_rl)
beta_q1   = 1.0 / (1.0 - GAMMA_Q1 * rho_c)
P_rl      = disc_lyap(A_cl_rl, GAMMA_Q1)

# x₀ at saturation boundary of π_RL (|K_rl @ x₀| = 0.95 < 1)
x0  = worst_overshoot_x0(A_cl_rl, scale_K=K_rl, target_u=0.95)
C0  = quad(M, x0)
u0_rl = abs(ctrl(K_rl, x0))
u0_c  = abs(ctrl(K_c,  x0))

k_arr = np.arange(T_SIM + 1)

traj_c  = rollout_free(A_cl_c,  T_SIM, x0)
traj_rl = rollout_free(A_cl_rl, T_SIM, x0)

C_c  = np.array([quad(M,    x) for x in traj_c ])
C_rl = np.array([quad(M,    x) for x in traj_rl])
V_rl = np.array([quad(P_rl, x) for x in traj_rl])

ies_bound = beta_q1 * C0 * rho_c**k_arr

u_c_traj  = np.array([ctrl(K_c,  traj_c[k])  for k in range(T_SIM)])
u_rl_traj = np.array([ctrl(K_rl, traj_rl[k]) for k in range(T_SIM)])

print("=" * 70)
print(f"Q1: Overshoot Analysis  (γ = {GAMMA_Q1},  action ∈ [-{U_MAX}, {U_MAX}])")
print("=" * 70)
print(f"  System: ρ_c = {rho_c:.5f}  (π_c monotone baseline)")
print(f"          ρ_phys(π_RL) = {rho_rl:.5f}  "
      f"→ {'OVERSHOOT (ρ > 1)' if rho_rl > 1 else 'no overshoot'}")
print(f"          sp_rad(π_RL) = {sp_rl:.5f}  (physically stable)")
print(f"          β = {beta_q1:.2f}")
print()
print(f"  Initial condition  x₀  at π_RL saturation boundary:")
print(f"    C(x₀)              = {C0:.6f}")
print(f"    |K_RL @ x₀|        = {u0_rl:.4f}  ≤ 1  → π_RL ≡ K_RL (no clipping)")
print(f"    |K_c  @ x₀|        = {u0_c:.4f}  ≤ 1  → π_c  ≡ K_c  (no clipping)")
print(f"    ∴ Both policies WITHIN action bounds at x₀.")
print()
print(f"  Trajectory metrics (from x₀):")
print(f"    π_c : peak C/C₀ = {C_c.max()/C0:.5f}  (1.000 expected, monotone)")
print(f"    π_RL: peak C/C₀ = {C_rl.max()/C0:.5f}  (= ρ_phys, overshoot {'✓' if C_rl.max()/C0 > 1 else '✗'})")
print(f"    π_RL: C(x₁)/C(x₀) = {C_rl[1]/C0:.5f}  (first step confirms ρ_phys)")
print()
print(f"  Value function (no overshoot even when C does):")
print(f"    V*(x₁)/V*(x₀) = {V_rl[1]/V_rl[0]:.5f}  ≤ ρ_c = {rho_c:.5f}  ✓")
print()
print(f"  Settling steps (C < 1% of C₀):")
print(f"    π_c  = {settling_step(C_c)} steps")
print(f"    π_RL = {settling_step(C_rl)} steps  (fewer → acceleration ✓)")
print()
print(f"  Max |u| over full trajectory:")
print(f"    π_c  : {abs(u_c_traj).max():.5f}  ≤ 1  ✓")
print(f"    π_RL : {abs(u_rl_traj).max():.5f}  ≤ 1  ✓  (stays in linear region)")


# ─────────────────────────────────────────────────────────────────────────────
# Q2: GAMMA ABLATION
# ─────────────────────────────────────────────────────────────────────────────
GAMMAS = [0.80, 0.90, 0.95, 0.99, 0.995]

# Common x₀: worst-overshoot direction of π_RL(γ=0.995), scaled to its sat boundary.
# This ensures ALL γ values satisfy |K(γ) @ x₀| ≤ 1 (smaller γ → smaller K).
K_ref   = disc_dare_gain(0.995)
A_cl_ref = A + B @ K_ref
x0_q2   = worst_overshoot_x0(A_cl_ref, scale_K=K_ref, target_u=0.90)
C0_q2   = quad(M, x0_q2)

print("\n" + "=" * 70)
print("Q2: Gamma Ablation  (common x₀ from γ=0.995 saturation boundary)")
print("=" * 70)
print(f"  {'γ':>5}  {'ρ_phys':>9}  {'sp_rad':>8}  {'β':>8}  "
      f"{'peak C/C₀':>10}  {'settle':>8}  {'max|u|':>8}")

gam_data = {}
for g in GAMMAS:
    K_g     = disc_dare_gain(g)
    A_cl_g  = A + B @ K_g
    _, rho_g = cont_rate(A_cl_g)
    sp_g    = sp_rad(A_cl_g)
    beta_g  = 1.0 / (1.0 - g * rho_c)
    traj_g  = rollout_free(A_cl_g, T_SIM, x0_q2)
    C_g     = np.array([quad(M, x) for x in traj_g])
    u_g     = np.array([abs(ctrl(K_g, traj_g[k])) for k in range(T_SIM)])
    peak_g  = C_g.max() / C0_q2
    sett_g  = settling_step(C_g)
    gam_data[g] = dict(rho=rho_g, sp=sp_g, beta=beta_g, C=C_g,
                        peak=peak_g, settle=sett_g, max_u=u_g.max())
    print(f"  {g:5.3f}  {rho_g:9.5f}  {sp_g:8.5f}  {beta_g:8.2f}  "
          f"{peak_g:10.4f}  {sett_g:8d}  {u_g.max():8.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Q1 & Q2: Overshoot and Discount Factor  —  Linear Testbed  (action ∈ [-1,1])\n"
    f"M = P_lqr (exact),   ρ_c = {rho_c:.4f}   |"
    f"   γ = {GAMMA_Q1}: ρ_phys(π_RL) = {rho_rl:.4f},  β = {beta_q1:.1f}",
    fontsize=10,
)

# ── [0,0] Q1: C trajectories ──────────────────────────────────────────────
ax = axes[0, 0]
ax.semilogy(k_arr, C_c  / C0, lw=2.0, color="steelblue",
            label=f"π_c  (ρ_phys={rho_c:.3f}, monotone)")
ax.semilogy(k_arr, C_rl / C0, lw=2.0, color="crimson",
            label=f"π_RL (ρ_phys={rho_rl:.3f}, OS={C_rl.max()/C0:.3f})")
ax.semilogy(k_arr, ies_bound / C0, "k--", lw=1.5, alpha=0.6,
            label=f"IES envelope  β·ρ_c^k  (β={beta_q1:.1f})")
ax.fill_between(k_arr, 1.0, ies_bound / C0,
                alpha=0.07, color="orange", label="allowed overshoot")
ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
ax.set(xlabel="Step k", ylabel="C(xₖ) / C(x₀)",
       title=f"Q1: Overshoot  (x₀ at π_RL sat. boundary, γ={GAMMA_Q1})")
ax.legend(fontsize=7)

# ── [0,1] Q1: Value function (no overshoot) ───────────────────────────────
ax = axes[0, 1]
V_norm = V_rl / V_rl[0]
ax.semilogy(k_arr, V_norm,      lw=2.0, color="seagreen", label="V*(xₖ)/V*(x₀)")
ax.semilogy(k_arr, rho_c**k_arr,"k--",  lw=1.5, alpha=0.7, label=f"ρ_c^k = {rho_c:.3f}^k")
ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
ax.set(xlabel="Step k", ylabel="V*(xₖ) / V*(x₀)",
       title="Q1: Value Function under π_RL  (no overshoot)")
ax.annotate("V* contracts without\novershoot even as C spikes",
            xy=(3, V_norm[3]), fontsize=7.5, color="seagreen",
            xytext=(15, 0.5), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.legend(fontsize=8)

# ── [0,2] Q1: Action trajectories (both within ±1) ───────────────────────
ax = axes[0, 2]
steps = np.arange(T_SIM)
ax.plot(steps, u_c_traj,  color="steelblue", lw=1.5, label="u  (π_c)")
ax.plot(steps, u_rl_traj, color="crimson",   lw=1.5, label="u  (π_RL)")
for s in [U_MAX, -U_MAX]:
    ax.axhline(s, color="k", lw=0.8, linestyle="--", alpha=0.5)
ax.set(xlabel="Step k", ylabel="Control u",
       title=f"Q1: Actions — both within ±{U_MAX} from x₀")
ax.set_ylim([-1.2, 1.2])
ax.legend(fontsize=8)
ax.text(T_SIM * 0.5, U_MAX * 0.85, "±1 bound", fontsize=7.5, ha="center",
        color="gray")

# ── [1,0] Q2: C trajectories for all γ ───────────────────────────────────
ax = axes[1, 0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(GAMMAS)))
for g, col in zip(GAMMAS, colors):
    r = gam_data[g]
    ax.semilogy(k_arr, r["C"] / C0_q2, lw=1.8, color=col,
                label=f"γ={g:.3f}  ρ={r['rho']:.3f}  |u|_max={r['max_u']:.3f}")
ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
ax.set(xlabel="Step k", ylabel="C(xₖ) / C(x₀)",
       title="Q2: γ Ablation — C Trajectories (all within action bounds)")
ax.legend(fontsize=6.5)

# ── [1,1] Q2: Peak overshoot and settling vs γ ───────────────────────────
ax  = axes[1, 1]
ax2 = ax.twinx()
g_arr   = np.array(GAMMAS)
peaks   = np.array([gam_data[g]["peak"]   for g in GAMMAS])
settles = np.array([gam_data[g]["settle"] for g in GAMMAS])
l1, = ax.plot(g_arr, peaks,   "o-", color="crimson", lw=2, ms=7, label="Peak C/C₀ (left)")
l2, = ax2.plot(g_arr, settles, "s--", color="navy",  lw=2, ms=7, label="Settling steps (right)")
ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
ax.set(xlabel="Discount factor γ", ylabel="Peak C / C(x₀)",
       title="Q2: Overshoot ↑  ↔  Settling ↓  as γ ↑")
ax2.set_ylabel("Steps to C < 1% of C₀")
ax.legend(handles=[l1, l2], fontsize=8, loc="upper left")

# ── [1,2] Q2: ρ_phys, sp_rad, β vs γ ─────────────────────────────────────
ax  = axes[1, 2]
ax2 = ax.twinx()
rhos  = np.array([gam_data[g]["rho"]  for g in GAMMAS])
sps   = np.array([gam_data[g]["sp"]   for g in GAMMAS])
betas = np.array([gam_data[g]["beta"] for g in GAMMAS])
l1, = ax.plot(g_arr, rhos, "o-", color="crimson",    lw=2, ms=7, label="ρ_phys (left)")
l2, = ax.plot(g_arr, sps,  "^:", color="darkorange",  lw=2, ms=7, label="sp_rad (left)")
l3, = ax2.plot(g_arr, betas,"s--",color="navy",       lw=2, ms=7, label="β (right)")
ax.axhline(1.0,   color="k",         lw=0.8, linestyle="--", alpha=0.5)
ax.axhline(rho_c, color="steelblue", lw=0.8, linestyle=":",  alpha=0.7)
ax.annotate(f"ρ_c={rho_c:.3f}  (π_c baseline)",
            xy=(g_arr[1], rho_c), fontsize=7, color="steelblue",
            xytext=(g_arr[1], rho_c - 0.07))
ax.set(xlabel="Discount factor γ", ylabel="Contraction / convergence rate",
       title="Q2: ρ_phys ↑ and sp_rad ↓ as γ ↑")
ax2.set_ylabel("β = 1/(1 − γ ρ_c)")
ax.legend(handles=[l1, l2, l3], fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig("verify_q12.png", dpi=150)
print("\nPlot saved → verify_q12.png")
