import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from policy.base import Base
from policy.cmg_pretrain import SDLQRPretrainMixin


class C3M(SDLQRPretrainMixin, Base):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        CMG: nn.Module,
        actor: nn.Module,
        data: dict,
        get_f_and_B: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        W_tol: float = 1e-5,  # Minimum loss improvement to be considered "significant"
        W_patience: int = 20,  # Number of updates to wait before stopping
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        warmup_epochs: int = 0,
        # --- optional SD-LQR CMG pretraining (the CORL recipe) ---
        pretrain_cmg: bool = False,
        SDC_func: nn.Module = None,
        Q_scaler: float = 1.0,
        R_scaler: float = 0.0,
        pretrain_epochs: int = 5000,
        pretrain_buffer_size: int = 10000,
        pretrain_minibatch_size: int = 1024,
        pretrain_log_interval: int = 100,
        pretrain_W_lr: float = 1e-3,
        val_split: float = 0.1,
        val_batch_size: int = 512,
        val_interval: int = 25,
        plateau_window: int = 5,
        plateau_tol: float = 1e-3,
        plateau_patience: int = 3,
        restore_best: bool = True,
        logger=None,
        writer=None,
        device: str = "cpu",
    ):
        super(C3M, self).__init__()

        # constants
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates

        # trainable networks
        self.CMG = CMG
        self.actor = actor

        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

        self.eps = eps
        self.W_tol = W_tol
        self.W_patience = W_patience
        self.w_ub = w_ub
        self.w_lb = w_lb
        self.lbd = lbd

        self.W_patience_counter = 0
        self.best_W_loss = float("inf")
        self.stop_W_training = False  # Flag to freeze W updates

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.CMG.parameters(), "lr": W_lr},
                {"params": self.actor.parameters(), "lr": u_lr},
            ]
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            self.warmup_W()

        # === OPTIONAL: SD-LQR CMG PRETRAINING (the CORL recipe) === #
        # Warm-starts the contraction metric by minimizing only the contraction loss
        # against SD-LQR controls (no c1/c2). Unlike CORL we keep the CMG trainable
        # afterwards, so the joint C3M objective refines it from the pretrained init.
        self.pretrain_cmg = pretrain_cmg
        if self.pretrain_cmg:
            if SDC_func is None:
                raise ValueError(
                    "C3M CMG pretraining requires a trained SDC_func (set --c3m-pretrain-cmg "
                    "so the SDC decomposition network is learned first)."
                )

            # SD-LQR ingredients + contraction constants used by the pretrain loss.
            self.SDC_func = SDC_func.eval()
            self.Q_scaler = Q_scaler
            self.R_scaler = R_scaler
            self.W_entropy_scaler = 0.0  # deterministic CMG -> no entropy term

            # pretraining configuration
            self.pretrain_epochs = pretrain_epochs
            self.pretrain_buffer_size = pretrain_buffer_size
            self.pretrain_minibatch_size = pretrain_minibatch_size
            self.pretrain_log_interval = pretrain_log_interval
            self.pretrain_W_lr = pretrain_W_lr

            # early stopping: validation total-loss, plateau-slope rule, restore best
            self.val_split = val_split
            self.val_batch_size = val_batch_size
            self.val_interval = val_interval
            self.plateau_window = plateau_window
            self.plateau_tol = plateau_tol
            self.plateau_patience = plateau_patience
            self.restore_best = restore_best

            self.logger = logger
            self.writer = writer

            # Pretrain the CMG with a dedicated optimizer, then FREEZE it: the
            # SD-LQR-warm-started metric defines the contraction geometry, and
            # policy synthesis (the C3M actor) is trained against this fixed metric
            # rather than co-adapting it.
            self.cmg_pretrain_optimizer = torch.optim.Adam(
                self.CMG.parameters(), lr=pretrain_W_lr
            )
            self.freeze_cmg_after_pretrain = True

            self.pretrain_CMG()

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)



    def forward(self, state: np.ndarray):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        a, _ = self.actor(state)

        return a, {
            "probs": self.dummy,  # dummy for code consistency
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def compute_loss(self, warming_up: bool = False):
        #
        I = torch.eye(self.x_dim, device=self.device)

        # === SAMPLE BATCH === #
        batch = dict()
        buffer_size, batch_size = self.data["x"].shape[0], 1024
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        for key in self.data.keys():
            # Sample a batch of 1024
            batch[key] = self.data[key][indices]

        # === PREPARE TENSORS === #
        x = self.to_tensor(batch["x"]).requires_grad_()
        xref = self.to_tensor(batch["xref"])
        uref = self.to_tensor(batch["uref"])

        raw_W, _ = self.CMG(x)  # n, x_dim, x_dim
        # Lower-bound the metric inverse (no-op for the bounded CMG, which is
        # already SPD in (w_lb, w_ub) by construction).
        W = self._bound_W(raw_W)
        # linalg.solve has better-conditioned gradients than inverse() when W is near-singular
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        f, B, Bbot = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)  # n, x_dim
        B = B.to(self._dtype).to(self.device)  # n, x_dim, action
        Bbot = Bbot.to(self._dtype).to(self.device)  #

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim

        f = f.detach()
        B = B.detach()
        Bbot = Bbot.detach()

        # since online we do not do below
        state = torch.concatenate([x, xref, uref], dim=1)
        u, _ = self.actor(state)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.u_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        # contraction condition
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))
        # this has to be a negative definite matrix
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.u_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
            C2_inner = DbW - 2 * sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)

            C2_inners.append(C2_inner)
            C2s.append(C2)

        ### DEFINE PD MATRICES ###
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)
        C1 = C1 + self.eps * torch.eye(C1.shape[-1], device=self.device)
        C2 = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        overshoot = W - self.w_ub * I

        # === DEFINE LOSSES === #
        pd_loss, pd_reg = self.loss_pos_matrix_random_sampling(-Cu)
        c1_loss, c1_reg = self.loss_pos_matrix_random_sampling(-C1)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_random_sampling(-overshoot)
        c2_loss = C2

        if warming_up:
            loss = c1_loss + c2_loss + c1_reg + overshoot_loss + overshoot_reg
        else:
            self.record_eigenvalues(Cu, dot_M, sym_MABK, C1, C2, overshoot)
            loss = (
                overshoot_loss
                + pd_loss
                + c1_loss
                + c2_loss
                + pd_reg
                + c1_reg
                + overshoot_reg
            )

        return (
            loss,
            {
                "pd_loss": pd_loss,
                "c1_loss": c1_loss,
                "c2_loss": c2_loss,
                "overshoot_loss": overshoot_loss,
            },
        )

    def optimize_params(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()

        # NaN/Inf gradients (e.g. from near-singular W inverse) corrupt weights even
        # after clipping — detect them early and skip this step entirely.
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in self.parameters()
        ):
            self.optimizer.zero_grad()
            return {}

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.CMG, self.actor, self.lbd],
            ["CMG", "actor", "lbd"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        return grad_dict

    def warmup_W(self):
        # Configuration
        max_epochs = self.warmup_epochs

        prev_loss = float("inf")
        stagnant_epochs = 0

        with tqdm(range(max_epochs), desc="Warmup Phase") as pbar:
            for epoch in pbar:
                # 1. Train Step
                loss, infos = self.compute_loss(warming_up=True)
                self.optimize_params(loss)

                # 2. Get scalar loss
                current_loss = loss.item() if hasattr(loss, "item") else loss

                # 3. Calculate Improvement
                loss_change = prev_loss - current_loss

                # 4. Update Progress Bar
                pbar.set_postfix(
                    loss=f"{current_loss:.4f}", change=f"{loss_change:.1e}"
                )

                # 5. Convergence Check (Minimal Change)
                # We check if improvement is positive but very small
                if 0 <= loss_change < self.W_tol:
                    stagnant_epochs += 1
                else:
                    stagnant_epochs = 0  # Reset if we see good improvement or a spike

                # Check if we have been stagnant for 'patience' epochs
                if stagnant_epochs >= self.W_patience:
                    pbar.write(
                        f"✓ Warmup converged: Loss stabilized at {current_loss:.4f} (Change < {self.W_tol} for {self.W_patience} epochs)"
                    )
                    break

                prev_loss = current_loss

            else:
                pbar.write(
                    f"⚠ Max warmup epochs ({max_epochs}) reached without full stabilization."
                )

    def learn(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # === PERFORM OPTIMIZATION STEP === #
        loss, infos = self.compute_loss()
        grad_dict = self.optimize_params(loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_updates % 500 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["C3M/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/lr/W_lr": self.lr_scheduler.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.lr_scheduler.get_last_lr()[1],
        }
        norm_dict = self.compute_weight_norm(
            [self.CMG, self.actor],
            ["CMG", "actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # === CLEANUP === #
        self.eval()
        update_time = time.time() - t0
        self.num_updates += 1

        return loss_dict, supp_dict, update_time
