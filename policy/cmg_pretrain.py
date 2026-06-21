"""Reusable SD-LQR CMG pretraining (the CORL recipe).

The contraction metric generator (CMG) is pretrained by minimizing only the
contraction loss Cu, where the control used inside the contraction condition is
computed by SD-LQR: for each sampled state we solve the state-dependent Riccati
equation to obtain u = SD-LQR(x) and the feedback gain K_lqr, and use (u, -K_lqr)
in the contraction condition. The c1/c2 conditions are dropped during pretraining.

This logic was originally written for CORL; it is factored into ``SDLQRPretrainMixin``
so that other policies (e.g. C3M) can warm-start their CMG with the same recipe.

A host class that mixes this in must provide, before calling :meth:`pretrain_CMG`:

  * networks/dynamics: ``CMG``, ``SDC_func`` (eval mode), ``get_f_and_B``
  * an optimizer over the CMG parameters: ``cmg_pretrain_optimizer``
  * geometry: ``x_dim``, ``u_dim``, ``device``, ``_dtype``
  * an offline dataset dict ``data`` with keys ``x``, ``xref``, ``uref``
  * contraction constants: ``w_lb``, ``w_ub``, ``lbd``, ``eps``, ``W_entropy_scaler``
  * SD-LQR weights: ``Q_scaler``, ``R_scaler``
  * pretraining config: ``pretrain_epochs``, ``pretrain_buffer_size``,
    ``pretrain_minibatch_size``, ``pretrain_log_interval``, ``pretrain_W_lr``,
    ``val_split``, ``val_batch_size``, ``val_interval``, ``plateau_window``,
    ``plateau_tol``, ``plateau_patience``, ``restore_best``,
    ``freeze_cmg_after_pretrain``
  * logging handles: ``logger``, ``writer`` (either may be ``None``)
  * the Base/Utilities helpers ``to_tensor``, ``Jacobian``, ``B_Jacobian``,
    ``weighted_gradients``, ``loss_pos_matrix_eigen``, ``compute_gradient_norm``

It also relies on ``name`` for log-tag namespacing and sets ``warmup_result``
(a summary figure, or ``None`` when streamed to wandb) and ``stop_W_training``.
"""

import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch import matmul, transpose
from tqdm import tqdm


class SDLQRPretrainMixin:
    # ------------------------------------------------------------------ #
    # SD-LQR batched controls + feedback gains
    # ------------------------------------------------------------------ #
    def _sdlqr_controls(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor):
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
        """Precompute SD-LQR controls/gains once for a fixed pretraining buffer.

        All tensors are kept on CPU and moved to device per-minibatch in
        compute_pretrain_loss. This avoids pinning n × x_dim² gain tensors in
        VRAM for the full pretraining horizon — critical when multiple agents share
        one GPU with large CMG architectures (width up to 1024).
        """
        buffer_size = self.data["x"].shape[0]
        n = min(self.pretrain_buffer_size, buffer_size)
        indices = np.random.choice(buffer_size, size=n, replace=False)

        # Keep on CPU; only move to device per-chunk for the Riccati solve.
        x = torch.from_numpy(self.data["x"][indices]).to(self._dtype)
        xref = torch.from_numpy(self.data["xref"][indices]).to(self._dtype)
        uref = torch.from_numpy(self.data["uref"][indices]).to(self._dtype)

        # Solve the per-sample Riccati equations once, in chunks.
        chunk = 2048
        u_list, K_list = [], []
        for s in tqdm(range(0, n, chunk), desc=f"{self.name} SD-LQR pretrain buffer"):
            sl = slice(s, min(s + chunk, n))
            u_c, K_c = self._sdlqr_controls(
                x[sl].to(self.device), xref[sl].to(self.device), uref[sl].to(self.device)
            )
            u_list.append(u_c.cpu())
            K_list.append(K_c.cpu())

        self.pretrain_data = {
            "x": x,
            "xref": xref,
            "uref": uref,
            "u": torch.cat(u_list, dim=0),  # (n, u_dim), CPU
            "K": torch.cat(K_list, dim=0),  # (n, u_dim, x_dim), CPU
        }
        self.pretrain_size = n

        # Hold out an in-distribution validation split for early stopping.
        # Keep index tensors on CPU too — they are only used for indexing.
        perm = torch.randperm(n)
        n_val = max(1, int(self.val_split * n))
        self.val_indices = perm[:n_val]
        self.train_indices = perm[n_val:]
        if len(self.train_indices) == 0:  # tiny buffers: fall back to shared set
            self.train_indices = perm
        v = min(n_val, self.val_batch_size)
        vb = self.val_indices[:v]
        self.val_batch = {k: self.pretrain_data[k][vb] for k in self.pretrain_data}

    # ------------------------------------------------------------------ #
    # Pretraining: minimize only the contraction loss Cu (no c1/c2)
    # ------------------------------------------------------------------ #
    def compute_pretrain_loss(self, batch: dict, create_graph: bool = True):
        """Compute the CMG pretraining loss.

        Pass create_graph=False for validation checks: the loss value is identical
        but the Jacobian graphs are not retained for a second backward pass, saving
        x_dim² × CMG_activation_size of VRAM per validation call.
        """
        I = torch.eye(self.x_dim, device=self.device)

        # Move CPU-resident buffer tensors to device for this minibatch.
        x = batch["x"].to(self.device).clone().requires_grad_()
        u = batch["u"].to(self.device)   # precomputed SD-LQR control (detached)
        K = batch["K"].to(self.device)   # precomputed differential gain -K_lqr (detached)

        raw_W, info_W = self.CMG(x)  # n, x_dim, x_dim
        # Lower-bound the metric inverse (no-op for the bounded CMG, which is
        # already SPD in (w_lb, w_ub) by construction).
        W = self._bound_W(raw_W)
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        # === DYNAMICS (known) === #
        f, B, _ = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)
        B = B.to(self._dtype).to(self.device)

        DfDx = self.Jacobian(f, x, create_graph=create_graph)  # n, x_dim, x_dim
        DBDx = self.B_Jacobian(B, x, create_graph=create_graph)  # n, x_dim, x_dim, u_dim

        f = f.detach()
        B = B.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.u_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, create_graph=create_graph)

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
            cu_eig = torch.linalg.eigvalsh(Cu.cpu() if Cu.device.type == "mps" else Cu)
            w_eig = torch.linalg.eigvalsh(W.cpu() if W.device.type == "mps" else W)
            # Cu must be negative definite for contraction: track its max eigenvalue.
            cu_max_eig = cu_eig.max(dim=-1).values.mean().item()
            # condition number of the metric (overshoot/tube proxy).
            W_cond = (w_eig[:, -1] / w_eig[:, 0].clamp_min(1e-8)).mean().item()

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

    def _optimize_cmg_pretrain(self, loss: torch.Tensor) -> float:
        """One optimizer step over the CMG parameters; returns the CMG grad norm."""
        opt = self.cmg_pretrain_optimizer
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.CMG.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.CMG], ["CMG"], dir=self.name, device=self.device
        )
        opt.step()
        return grad_dict.get(f"{self.name}/grad/CMG", 0.0)

    def _init_pretrain_state(self):
        """Set up the history buffers used during pretraining/logging."""
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
        self.val_history = {"epoch": [], "val_loss": []}

    def _log_pretrain_step(self, epoch: int, current_loss: float, infos: dict,
                           grad_norm: float):
        """Record per-step pretraining metrics to history, TensorBoard and console."""
        W_lr = self.cmg_pretrain_optimizer.param_groups[0]["lr"]
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

        # Real-time wandb curves against a custom pretrain-epoch x-axis. Downsampled
        # so the number of global-step increments stays small (see _setup_wandb_pretrain).
        if getattr(self, "_wandb_active", False) and (
            epoch % self._wandb_interval == 0 or epoch == self.pretrain_epochs - 1
        ):
            import wandb

            payload = {
                f"{self.name}/pretrain/{k}": v
                for k, v in record.items()
                if k != "epoch"
            }
            payload[f"{self.name}/pretrain/epoch"] = epoch
            wandb.log(payload)

    def _log_validation(self, epoch: int, val_loss: float):
        """Log the held-out validation loss to TensorBoard and wandb."""
        if self.writer is not None:
            self.writer.add_scalar(f"{self.name}/pretrain/val_loss", val_loss, epoch)
        if getattr(self, "_wandb_active", False):
            import wandb

            wandb.log(
                {
                    f"{self.name}/pretrain/val_loss": val_loss,
                    f"{self.name}/pretrain/epoch": epoch,
                }
            )

    def _setup_wandb_pretrain(self):
        """Register a custom epoch x-axis so pretraining curves stream live to wandb.

        wandb shares one monotonically-increasing global step across the whole run
        (SDC training -> pretraining -> RL). We therefore (a) plot pretrain metrics
        against their own ``pretrain/epoch`` axis via define_metric, and (b) downsample
        so pretraining adds only a few hundred global-step increments, keeping it from
        clobbering the trainer's step sequence.
        """
        self._wandb_active = False
        try:
            import wandb

            if wandb.run is None:
                return
            wandb.define_metric(f"{self.name}/pretrain/epoch")
            wandb.define_metric(
                f"{self.name}/pretrain/*", step_metric=f"{self.name}/pretrain/epoch"
            )
            max_points = 300
            self._wandb_interval = max(1, self.pretrain_epochs // max_points)
            self._wandb_active = True
        except Exception as exc:  # pragma: no cover
            print(f"[{self.name}] wandb pretrain logging unavailable: {exc}")

    def _validation_loss(self):
        """Total pretrain loss on the held-out (in-distribution) validation batch.

        Uses create_graph=False: we only need the numerical loss value, not its
        gradient w.r.t. CMG parameters. This avoids retaining x_dim² second-order
        graphs (up to ~256 MB for CMG width=1024) on every validation check.
        """
        was_training = self.training
        self.eval()
        loss, _ = self.compute_pretrain_loss(self.val_batch, create_graph=False)
        val = loss.item()
        if was_training:
            self.train()
        return val

    def _plateau_reached(self):
        """Plateau-slope rule: True once the moving average of the validation loss
        flattens (relative change < tol) for ``plateau_patience`` consecutive checks."""
        vals = self.val_history["val_loss"]
        w = self.plateau_window
        if len(vals) < 2 * w:  # need two full windows
            return False
        ma_now = float(np.mean(vals[-w:]))
        ma_prev = float(np.mean(vals[-2 * w : -w]))
        rel = abs(ma_now - ma_prev) / (abs(ma_prev) + 1e-12)
        if rel < self.plateau_tol:
            self._plateau_count = getattr(self, "_plateau_count", 0) + 1
        else:
            self._plateau_count = 0
        return self._plateau_count >= self.plateau_patience

    def _snapshot_best(self):
        return {k: v.detach().clone() for k, v in self.CMG.state_dict().items()}

    def pretrain_CMG(self):
        """Pretrain the CMG with the contraction loss formed from SD-LQR controls.

        Early stopping: monitor the validation total-loss on a held-out split; stop
        when its moving-average slope plateaus; restore the best-validation weights.
        """
        self._init_pretrain_state()
        self._build_pretrain_buffer()
        self._setup_wandb_pretrain()

        # Start the CMG optimizer at pretrain_W_lr; we cosine-anneal it to 0 over the
        # pretraining horizon (set per epoch below).
        for g in self.cmg_pretrain_optimizer.param_groups:
            g["lr"] = self.pretrain_W_lr

        best_val = float("inf")
        best_state = self._snapshot_best()
        best_epoch = 0
        self._plateau_count = 0
        stopped_reason = f"reached max epochs ({self.pretrain_epochs})"
        self.train()

        with tqdm(range(self.pretrain_epochs), desc=f"{self.name} CMG Pretrain") as pbar:
            for epoch in pbar:
                # cosine LR anneal: pretrain_W_lr -> 0 across the pretraining horizon
                frac = epoch / max(1, self.pretrain_epochs)
                lr = self.pretrain_W_lr * 0.5 * (1.0 + np.cos(np.pi * frac))
                for g in self.cmg_pretrain_optimizer.param_groups:
                    g["lr"] = lr

                # sample a minibatch from the TRAIN split of the SD-LQR buffer
                mb = min(self.pretrain_minibatch_size, len(self.train_indices))
                sel = torch.randperm(len(self.train_indices))[:mb]  # CPU randperm
                idx = self.train_indices[sel]
                batch = {k: v[idx] for k, v in self.pretrain_data.items()}

                loss, infos = self.compute_pretrain_loss(batch)
                grad_norm = self._optimize_cmg_pretrain(loss)

                current_loss = loss.item()
                self._log_pretrain_step(epoch, current_loss, infos, grad_norm)

                # --- validation + early-stopping check --- #
                if epoch % self.val_interval == 0 or epoch == self.pretrain_epochs - 1:
                    val_loss = self._validation_loss()
                    self.val_history["epoch"].append(epoch)
                    self.val_history["val_loss"].append(val_loss)
                    self._log_validation(epoch, val_loss)

                    if val_loss < best_val:
                        best_val = val_loss
                        best_epoch = epoch
                        if self.restore_best:
                            best_state = self._snapshot_best()

                    if self._plateau_reached():
                        stopped_reason = (
                            f"validation plateau at epoch {epoch} "
                            f"(val_loss={val_loss:.4f})"
                        )
                        pbar.write(f"✓ {self.name} pretrain early-stopped: {stopped_reason}")
                        break

                pbar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    val=f"{best_val:.4f}",
                    cu_max=f"{infos['cu_max_eig']:.2g}",
                )

                # periodic console summary for the log file
                if epoch % self.pretrain_log_interval == 0:
                    pbar.write(
                        f"[{self.name} pretrain {epoch}] loss={current_loss:.4f} "
                        f"best_val={best_val:.4f} "
                        f"pd={infos['pd_loss'].item():.3g} "
                        f"cu_max_eig={infos['cu_max_eig']:.3g} "
                        f"W_cond={infos['W_cond']:.3g} grad={grad_norm:.3g}"
                    )
            else:
                pbar.write(f"⚠ {self.name} pretrain {stopped_reason}.")

        # Restore the best-validation checkpoint.
        if self.restore_best:
            self.CMG.load_state_dict(best_state)
            print(
                f"[{self.name}] restored best CMG (val_loss={best_val:.4f} @ epoch {best_epoch})."
            )
        self.pretrain_summary = {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stopped_reason": stopped_reason,
        }

        self.eval()
        # Freeze the CMG so downstream policy synthesis trains against a fixed
        # contraction metric (both CORL and C3M). We disable grad on the CMG
        # parameters directly — C3M's main optimizer includes the CMG param group,
        # so the stop_W_training flag alone would not stop it from updating.
        if getattr(self, "freeze_cmg_after_pretrain", True):
            self.stop_W_training = True
            for p in self.CMG.parameters():
                p.requires_grad_(False)
        self._dump_pretrain_csv()

        fig = self._plot_pretrain_result()
        if getattr(self, "_wandb_active", False):
            # Log the summary figure ourselves (at the current valid global step) and
            # skip main.py's step-0 image re-log, which wandb would drop.
            import wandb

            wandb.log({f"{self.name}/pretrain/summary_curves": wandb.Image(fig)})
            self.warmup_result = None
        else:
            self.warmup_result = fig

        # Free the precomputed pretraining tensors. These live on the compute device
        # as plain attributes (not registered buffers), so leaving them around breaks
        # multiprocessing sampling on MPS/CUDA (the policy is pickled onto CPU workers
        # and stray device tensors cannot be shared). They are only needed here.
        del self.pretrain_data
        del self.val_batch
        del self.train_indices
        del self.val_indices

        # Drop the SDC network: it is only used to form SD-LQR controls during
        # pretraining, and its Adam optimizer keeps moment tensors on the compute
        # device (not moved by to_device), which would also break MP pickling.
        self.SDC_func = None

        # Drop the logger/writer handles: they hold wandb/file thread locks that are
        # not picklable, which breaks the multiprocessing sampler (the policy is
        # pickled onto workers). They are only used for pretraining logs.
        self.logger = None
        self.writer = None

        # Release cached VRAM immediately. Without this, PyTorch's CUDA caching
        # allocator holds onto the pretrain-phase memory even after del, leaving
        # less headroom for C3M main training — critical with 5 concurrent agents.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            axi.plot(e, h[key], label="train")
            axi.set_title(title)
            axi.set_xlabel("pretrain epoch")
            axi.grid(True, ls="--", alpha=0.5)
            if logy and len(h[key]) > 0 and min(h[key]) > 0:
                axi.set_yscale("log")
            if len(h[key]) > 0:
                axi.text(0.98, 0.95, f"{h[key][-1]:.3g}", ha="right", va="top",
                         transform=axi.transAxes, fontsize=9)
        # overlay held-out validation loss (early-stopping signal) on the loss panel
        if len(self.val_history["epoch"]) > 0:
            ax[0, 0].plot(
                self.val_history["epoch"], self.val_history["val_loss"],
                "r.-", label="val (early-stop)",
            )
            ax[0, 0].legend(fontsize=8)
            summ = getattr(self, "pretrain_summary", {})
            if "best_epoch" in summ:
                ax[0, 0].axvline(summ["best_epoch"], color="g", ls=":", lw=1)
        ax[1, 0].axhline(0.0, color="r", ls=":", lw=1)  # Cu target
        fig.suptitle(f"{self.name} CMG Pretraining (SD-LQR contraction loss)")
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

        path = os.path.join(log_dir, f"{self.name.lower()}_pretrain_history.csv")
        keys = list(self.pretrain_history.keys())
        n = len(self.pretrain_history["epoch"])
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for i in range(n):
                    writer.writerow([self.pretrain_history[k][i] for k in keys])
            print(f"[{self.name}] pretraining history written to {path}")
        except Exception as exc:
            print(f"[{self.name}] failed to write pretrain CSV: {exc}")
