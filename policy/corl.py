import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_continuous_are
from torch import inverse, matmul, transpose
from tqdm import tqdm

from policy.carl import CARL


class CORL(CARL):
    """Certified Optimal Nonlinear Control using RL (CORL).

    Differences from CARL:
      1. RL optimizes the *negative* error reward  r = -e^T M e  (not the inverse
         form used by CARL), normalized by a running average of its magnitude for
         numerical stability.
      2. The CMG is *pretrained* (not jointly trained) by minimizing only the
         contraction loss (Cu) -- the c1/c2 conditions used by CARL to guide an
         inaccurate jointly-trained policy are dropped. Instead, the control used
         inside the contraction loss is computed by SD-LQR: for each sampled state
         we solve the state-dependent Riccati equation to obtain u = SD-LQR(x) and
         the feedback gain K_lqr, and use (u, -K_lqr) in the contraction condition.
      3. After pretraining the CMG is frozen; RL only optimizes the contraction
         reward -e^T M e normalized by the running average.

    Assumes known dynamics (get_f_and_B is the true model).
    """

    def __init__(
        self,
        *args,
        SDC_func: nn.Module = None,
        Q_scaler: float = 1.0,
        R_scaler: float = 0.0,
        pretrain_epochs: int = 30000,
        pretrain_buffer_size: int = 10000,
        pretrain_minibatch_size: int = 1024,
        pretrain_patience: int = 200,
        pretrain_log_interval: int = 100,
        logger=None,
        writer=None,
        **kwargs,
    ):
        if SDC_func is None:
            raise ValueError("CORL requires a trained SDC_func for SD-LQR pretraining.")

        # CORL freezes the CMG during RL (pretrained, not jointly trained) and uses
        # the negative-error reward instead of CARL's inverse form.
        kwargs["disable_CMG_training"] = True
        kwargs["reward_mode"] = "default"
        kwargs["warmup_epochs"] = 0  # we run our own SD-LQR pretraining instead

        super().__init__(*args, **kwargs)

        self.name = "CORL"

        # SD-LQR ingredients
        self.SDC_func = SDC_func.eval()
        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler

        # pretraining configuration
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_buffer_size = pretrain_buffer_size
        self.pretrain_minibatch_size = pretrain_minibatch_size
        self.pretrain_patience = pretrain_patience
        self.pretrain_log_interval = pretrain_log_interval

        # logging handles (optional; set by get_policy). Pretraining happens in
        # __init__ before the trainer exists, so we log here directly.
        self.logger = logger
        self.writer = writer
        self.pretrain_history = {
            "epoch": [],
            "loss": [],
            "pd_loss": [],
            "overshoot_loss": [],
            "entropy_loss": [],
            "grad_norm": [],
            "cu_max_eig": [],
            "W_cond": [],
            "W_lr": [],
        }

        # running-average reward normalizer (EMA of |raw reward| magnitude)
        self.reward_norm_beta = 0.99
        self.running_reward_scale = None

        # Pretrain the contraction metric using SD-LQR controls.
        self.pretrain_CMG()

    # ------------------------------------------------------------------ #
    # SD-LQR batched controls + feedback gains
    # ------------------------------------------------------------------ #
    def _sdlqr_controls(
        self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor
    ):
        """Compute u = SD-LQR(x) and the differential feedback gain K = du/dx = -K_lqr.

        scipy's continuous-time ARE solver runs per-sample, so the Riccati solve is
        looped over the batch. Returns (u, K) with u: (n, u_dim), K: (n, u_dim, x_dim).
        """
        n = x.shape[0]
        e = x - xref
        sdc_input = torch.cat((x, e), dim=-1)

        with torch.no_grad():
            _, B, _ = self.get_f_and_B(x)
            B = B.to(self._dtype).to(self.device)  # (n, x_dim, u_dim)

            Af, Bf = self.SDC_func(sdc_input)  # Af: (n,x,x), Bf: (n,u,x,x)
            # Bf_u = sum_i uref_i * Bf_i  -> (n, x, x)
            Bf_u = (uref.view(n, self.u_dim, 1, 1) * Bf).sum(dim=1)
            A = Af + Bf_u  # state-dependent closed-form A(x)

        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        Q = (self.Q_scaler + 1e-5) * np.eye(self.x_dim, dtype=np.float64)
        R = (self.R_scaler + 1e-5) * np.eye(self.u_dim, dtype=np.float64)
        Rinv = np.linalg.inv(R)

        K_np = np.zeros((n, self.u_dim, self.x_dim), dtype=np.float32)
        for i in range(n):
            try:
                P = solve_continuous_are(
                    A_np[i].astype(np.float64), B_np[i].astype(np.float64), Q, R
                )
                K_np[i] = (Rinv @ B_np[i].T.astype(np.float64) @ P).astype(np.float32)
            except Exception:
                # Riccati solve failed (e.g. uncontrollable linearization): zero gain.
                K_np[i] = 0.0

        K_lqr = torch.from_numpy(K_np).to(self._dtype).to(self.device)  # (n,u,x)
        u = uref - matmul(K_lqr, e.unsqueeze(-1)).squeeze(-1)  # (n, u_dim)
        K_diff = -K_lqr  # differential feedback gain du/dx
        return u, K_diff

    def _build_pretrain_buffer(self):
        """Precompute SD-LQR controls/gains once for a fixed pretraining buffer."""
        buffer_size = self.data["x"].shape[0]
        n = min(self.pretrain_buffer_size, buffer_size)
        indices = np.random.choice(buffer_size, size=n, replace=False)

        x = self.to_tensor(self.data["x"][indices])
        xref = self.to_tensor(self.data["xref"][indices])
        uref = self.to_tensor(self.data["uref"][indices])

        # Solve the per-sample Riccati equations once, in chunks.
        chunk = 2048
        u_list, K_list = [], []
        for s in tqdm(range(0, n, chunk), desc="CORL SD-LQR pretrain buffer"):
            sl = slice(s, min(s + chunk, n))
            u_c, K_c = self._sdlqr_controls(x[sl], xref[sl], uref[sl])
            u_list.append(u_c)
            K_list.append(K_c)

        self.pretrain_data = {
            "x": x,
            "xref": xref,
            "uref": uref,
            "u": torch.cat(u_list, dim=0),  # (n, u_dim)
            "K": torch.cat(K_list, dim=0),  # (n, u_dim, x_dim)
        }
        self.pretrain_size = n

    # ------------------------------------------------------------------ #
    # Pretraining: minimize only the contraction loss Cu (no c1/c2)
    # ------------------------------------------------------------------ #
    def compute_pretrain_loss(self, batch: dict):
        I = torch.eye(self.x_dim, device=self.device)

        x = batch["x"].clone().requires_grad_()
        u = batch["u"]  # precomputed SD-LQR control (detached)
        K = batch["K"]  # precomputed differential gain -K_lqr (detached)

        raw_W, info_W = self.CMG(x)  # n, x_dim, x_dim
        W = raw_W + self.w_lb * I
        M = inverse(W)

        # === DYNAMICS (known) === #
        f, B, _ = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)
        B = B.to(self._dtype).to(self.device)

        DfDx = self.Jacobian(f, x)  # n, x_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, u_dim

        f = f.detach()
        B = B.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.u_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        # contraction condition (closed loop under SD-LQR feedback)
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)

        overshoot = W - self.w_ub * I

        # === LOSSES (contraction + conditioning only) === #
        pd_loss, pd_reg = self.loss_pos_matrix_eigen(-Cu)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_eigen(-overshoot)
        entropy_loss = info_W["entropy"].mean() * self.W_entropy_scaler

        loss = pd_loss + pd_reg + overshoot_loss + overshoot_reg - entropy_loss

        # === diagnostics (no grad) for debugging contraction feasibility === #
        with torch.no_grad():
            cu_eig = torch.linalg.eigvalsh(
                Cu.cpu() if Cu.device.type == "mps" else Cu
            )
            w_eig = torch.linalg.eigvalsh(W.cpu() if W.device.type == "mps" else W)
            # Cu must be negative definite for contraction: track its max eigenvalue.
            cu_max_eig = cu_eig.max(dim=-1).values.mean().item()
            # condition number of the metric (overshoot/tube proxy).
            W_cond = (
                (w_eig[:, -1] / w_eig[:, 0].clamp_min(1e-8)).mean().item()
            )

        return (
            loss,
            {
                "pd_loss": pd_loss,
                "overshoot_loss": overshoot_loss,
                "entropy_loss": entropy_loss,
                "cu_max_eig": cu_max_eig,
                "W_cond": W_cond,
            },
        )

    def _log_pretrain_step(self, epoch: int, current_loss: float, infos: dict,
                           grad_norm: float):
        """Record per-step pretraining metrics to history, TensorBoard and console."""
        W_lr = self.W_lr_scheduler.get_last_lr()[0]
        record = {
            "epoch": epoch,
            "loss": current_loss,
            "pd_loss": infos["pd_loss"].item(),
            "overshoot_loss": infos["overshoot_loss"].item(),
            "entropy_loss": infos["entropy_loss"].item(),
            "grad_norm": grad_norm,
            "cu_max_eig": infos["cu_max_eig"],
            "W_cond": infos["W_cond"],
            "W_lr": W_lr,
        }
        for k, v in record.items():
            self.pretrain_history[k].append(v)

        # TensorBoard curves (separate tag namespace -> no step collision with RL).
        if self.writer is not None:
            for k, v in record.items():
                if k == "epoch":
                    continue
                self.writer.add_scalar(f"{self.name}/pretrain/{k}", v, epoch)

    def pretrain_CMG(self):
        """Pretrain the CMG with the contraction loss formed from SD-LQR controls."""
        self._build_pretrain_buffer()

        best_loss, stagnant_epochs = float("inf"), 0
        self.train()

        with tqdm(range(self.pretrain_epochs), desc="CORL CMG Pretrain") as pbar:
            for epoch in pbar:
                # sample a minibatch from the precomputed SD-LQR buffer
                mb = min(self.pretrain_minibatch_size, self.pretrain_size)
                idx = torch.randperm(self.pretrain_size, device=self.device)[:mb]
                batch = {k: v[idx] for k, v in self.pretrain_data.items()}

                loss, infos = self.compute_pretrain_loss(batch)
                grad_dict = self.optimize_W_params(loss)
                grad_norm = grad_dict.get("CARL/grad/CMG", 0.0)

                current_loss = loss.item()
                self._log_pretrain_step(epoch, current_loss, infos, grad_norm)

                pbar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    pd=f"{infos['pd_loss'].item():.3g}",
                    cu_max=f"{infos['cu_max_eig']:.2g}",
                )

                # periodic console summary for the log file
                if epoch % self.pretrain_log_interval == 0:
                    pbar.write(
                        f"[CORL pretrain {epoch}] loss={current_loss:.4f} "
                        f"pd={infos['pd_loss'].item():.3g} "
                        f"overshoot={infos['overshoot_loss'].item():.3g} "
                        f"cu_max_eig={infos['cu_max_eig']:.3g} "
                        f"W_cond={infos['W_cond']:.3g} grad={grad_norm:.3g}"
                    )

                if current_loss < best_loss:
                    best_loss = current_loss
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1

                if stagnant_epochs >= self.pretrain_patience:
                    pbar.write(
                        f"✓ CORL pretrain converged: loss {current_loss:.4f} "
                        f"stable for {self.pretrain_patience} epochs."
                    )
                    break
            else:
                pbar.write(
                    f"⚠ CORL pretrain reached max epochs ({self.pretrain_epochs})."
                )

        self.eval()
        # Freeze CMG: stop_W_training together with disable_CMG_training guarantees
        # the contraction metric is fixed during RL.
        self.stop_W_training = True
        self.warmup_result = self._plot_pretrain_result()
        self._dump_pretrain_csv()

        # Free the precomputed buffer to save memory.
        del self.pretrain_data

    def _plot_pretrain_result(self):
        """Build a diagnostic figure of the CMG pretraining curves."""
        import matplotlib.pyplot as plt

        h = self.pretrain_history
        e = h["epoch"]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        panels = [
            ("loss", "Total loss", True),
            ("pd_loss", "Contraction (pd) loss", True),
            ("overshoot_loss", "Overshoot loss", True),
            ("cu_max_eig", "Cu max eig (<0 = contracting)", False),
            ("W_cond", "W condition number", True),
            ("grad_norm", "CMG grad norm", True),
        ]
        for axi, (key, title, logy) in zip(ax.ravel(), panels):
            axi.plot(e, h[key])
            axi.set_title(title)
            axi.set_xlabel("pretrain epoch")
            axi.grid(True, ls="--", alpha=0.5)
            if logy and len(h[key]) > 0 and min(h[key]) > 0:
                axi.set_yscale("log")
            if len(h[key]) > 0:
                axi.text(0.98, 0.95, f"{h[key][-1]:.3g}", ha="right", va="top",
                         transform=axi.transAxes, fontsize=9)
        ax[1, 0].axhline(0.0, color="r", ls=":", lw=1)  # Cu target
        fig.suptitle("CORL CMG Pretraining (SD-LQR contraction loss)")
        plt.tight_layout()
        plt.close(fig)
        return fig

    def _dump_pretrain_csv(self):
        """Write the pretraining history to the log directory for offline debug."""
        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return
        import csv
        import os

        path = os.path.join(log_dir, "corl_pretrain_history.csv")
        keys = list(self.pretrain_history.keys())
        n = len(self.pretrain_history["epoch"])
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for i in range(n):
                    writer.writerow([self.pretrain_history[k][i] for k in keys])
            print(f"[CORL] pretraining history written to {path}")
        except Exception as exc:
            print(f"[CORL] failed to write pretrain CSV: {exc}")

    # ------------------------------------------------------------------ #
    # Reward: negative-error form normalized by a running average
    # ------------------------------------------------------------------ #
    def get_rewards(self, states: torch.Tensor, controls: torch.Tensor):
        x, xref, uref = self.actor.trim_state(states)
        tracking_error = (x - xref).unsqueeze(-1)
        control_effort = torch.linalg.norm(controls, dim=-1, keepdim=True)

        with torch.no_grad():
            W, _ = self.CMG(x, deterministic=True)
            W = W + self.w_lb * torch.eye(self.x_dim, device=self.device).view(
                1, self.x_dim, self.x_dim
            )
            M = inverse(W)

            tracking_errorT = transpose(tracking_error, 1, 2)
            # Riemannian energy e^T M e  (>= 0)
            inner_quad = (tracking_errorT @ M @ tracking_error).squeeze(-1)

            # negative-error reward (NOT the inverse form used by CARL)
            raw_reward = (
                -self.tracking_scaler * inner_quad
                - self.control_scaler * control_effort
            )

        # === running-average normalization for numerical stability === #
        batch_scale = raw_reward.abs().mean().item()
        if self.running_reward_scale is None:
            self.running_reward_scale = batch_scale
        else:
            self.running_reward_scale = (
                self.reward_norm_beta * self.running_reward_scale
                + (1.0 - self.reward_norm_beta) * batch_scale
            )

        rewards = raw_reward / (self.running_reward_scale + 1e-8)

        ratio = self.running_reward_scale
        return rewards, ratio
