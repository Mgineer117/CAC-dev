import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from policy.cvstem import CVSTEM
from policy.layers.building_blocks import MLP, SirenNet
from policy.layers.policy_networks import get_activation


class NCM(Base):
    """Neural Contraction Metric (NCM), after Tsukamoto, Chung & Slotine
    (astrohiro/ncm; IEEE L-CSS 2021).

    Offline, CV-STEM solves a convex SDP at sampled states for the optimal
    contraction metrics M_k that minimize the steady-state tracking-error tube.
    The NCM is a neural network trained by SUPERVISED REGRESSION onto the Cholesky
    factors of those optimal metrics (MSE loss), exactly as in the reference. At
    runtime the metric drives the contraction controller

        u = u* - R^{-1} B(x)^T M(x) (x - x*),     M(x) = L(x) L(x)^T = NCM(x).

    Assumes known dynamics; A(x) = df/dx is used as the state-dependent coefficient.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        data: dict,
        get_f_and_B: Callable,
        dt: float,
        alpha: float = 0.5,
        w_nu: float = 1.0,
        R_scaler: float = 1.0,
        epsilon: float = 1e-4,
        linesearch: bool = True,
        include_dwdt: bool = True,
        hidden_dims: list = None,
        activation: str = "relu",
        w_lb: float = 1e-2,
        W_lr: float = 1e-3,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        cvstem_num_samples: int = 100,
        nupdates: int = 30000,
        num_windows: int = 1,
        device: str = "cpu",
        logger=None,
        writer=None,
    ):
        super(NCM, self).__init__()

        self.name = "NCM"
        self.device = device
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_windows = num_windows

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.nupdates = nupdates
        self.R_scaler = R_scaler
        self.w_lb = w_lb
        self.cvstem_num_samples = cvstem_num_samples

        self.logger = logger
        self.writer = writer

        # lower-triangular index map for the Cholesky vectorization
        self.tril_idx = torch.tril_indices(x_dim, x_dim, device=device)
        self.n_chol = x_dim * (x_dim + 1) // 2

        # --- metric network: x -> Cholesky vector of M(x) --- #
        if hidden_dims is None:
            hidden_dims = [128, 128]
        if str(activation).lower() == "siren":
            self.metric_net = SirenNet(
                input_dim=x_dim, hidden_dims=list(hidden_dims),
                output_dim=self.n_chol, device=device,
            )
        else:
            self.metric_net = MLP(
                x_dim, list(hidden_dims), self.n_chol,
                activation=get_activation(activation), device=device,
            )
        # alias so the shared C3MTrainer.save_model saves the metric net
        self.actor = self.metric_net

        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            self.get_f_and_B.eval()

        self.optimizer = torch.optim.Adam(
            [{"params": self.metric_net.parameters(), "lr": W_lr}]
        )
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_decay_lambda)

        self.cvstem = CVSTEM(
            x_dim=x_dim, u_dim=u_dim, dt=dt, alpha=alpha, w_nu=w_nu,
            epsilon=epsilon, linesearch=linesearch, include_dwdt=include_dwdt,
            device=device,
        )

        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

        self._build_cvstem_dataset()

    # ------------------------------------------------------------------ #
    # CV-STEM target generation
    # ------------------------------------------------------------------ #
    def _build_cvstem_dataset(self):
        buffer_size = self.data["x"].shape[0]
        n = min(self.cvstem_num_samples, buffer_size)
        indices = np.random.choice(buffer_size, size=n, replace=False)
        x_all = self.to_tensor(self.data["x"][indices])

        # A = df/dx (SDC) and B at the sampled states.
        x = x_all.clone().requires_grad_()
        f, B, _ = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device).reshape(n, self.x_dim)
        B = B.to(self._dtype).to(self.device).reshape(n, self.x_dim, self.u_dim)
        A = self.Jacobian(f, x).detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()

        print(f"[NCM] solving CV-STEM SDP over {n} samples ...")
        Ws, info = self.cvstem.solve(A, B_np)  # Ws: (n, x, x), dual metrics
        self.cvstem_info = info
        print(
            f"[NCM] CV-STEM done: status={info['status']}, alpha={info['alpha']:.3g}, "
            f"chi={info['chi']:.4g}, nu={info['nu']:.4g}, J={info['J']:.4g}"
        )

        # Optimal metrics M_k = W_k^{-1}; regress their Cholesky factors.
        W = torch.from_numpy(Ws).to(self._dtype).to(self.device)
        M = torch.inverse(W)
        M = 0.5 * (M + transpose(M, 1, 2))  # symmetrize
        L = torch.linalg.cholesky(M)  # lower-triangular, M = L L^T
        chol_target = L[:, self.tril_idx[0], self.tril_idx[1]]  # (n, n_chol)

        self.cvstem_data = {"x": x_all.detach(), "chol": chol_target.detach()}
        self.cvstem_size = n
        self.warmup_result = self._plot_cvstem_diagnostics(M.detach())

    # ------------------------------------------------------------------ #
    # Metric reconstruction + controller
    # ------------------------------------------------------------------ #
    def _vec_to_M(self, vec: torch.Tensor):
        """Reconstruct M = L L^T (+ w_lb I) from a batch of Cholesky vectors."""
        b = vec.shape[0]
        L = torch.zeros(b, self.x_dim, self.x_dim, device=self.device, dtype=self._dtype)
        L[:, self.tril_idx[0], self.tril_idx[1]] = vec
        M = matmul(L, transpose(L, 1, 2))
        M = M + self.w_lb * torch.eye(self.x_dim, device=self.device).unsqueeze(0)
        return M

    def trim_state(self, state: torch.Tensor):
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[
            :,
            (1 + self.num_windows) * self.x_dim : (1 + self.num_windows) * self.x_dim
            + self.u_dim,
        ]
        return x, xref, uref

    def forward(self, state: np.ndarray):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref = self.trim_state(state)
        e = (x - xref).unsqueeze(-1)  # (b, x_dim, 1)

        with torch.no_grad():
            vec = self.metric_net(x)
            M = self._vec_to_M(vec)

            _, B, _ = self.get_f_and_B(x)
            B = B.to(self._dtype).to(self.device).reshape(
                x.shape[0], self.x_dim, self.u_dim
            )
            Rinv = (1.0 / (self.R_scaler + 1e-6)) * torch.eye(
                self.u_dim, device=self.device
            )
            # u = u* - R^{-1} B^T M e
            BtMe = matmul(transpose(B, 1, 2), matmul(M, e))  # (b, u_dim, 1)
            u = uref - matmul(Rinv, BtMe).squeeze(-1)

        return u, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    # ------------------------------------------------------------------ #
    # Supervised regression onto Cholesky(M) (the NCM training)
    # ------------------------------------------------------------------ #
    def compute_loss(self):
        mb = min(self.minibatch_size, self.cvstem_size)
        idx = torch.randperm(self.cvstem_size, device=self.device)[:mb]
        x = self.cvstem_data["x"][idx]
        chol_target = self.cvstem_data["chol"][idx]

        chol_pred = self.metric_net(x)
        loss = self.mse_loss(chol_pred, chol_target)
        return loss

    def learn(self):
        self.train()
        t0 = time.time()

        # Drive the shared LR schedule from the update fraction.
        self.progress = min(1.0, self.num_updates / max(1, self.nupdates))

        loss = self.compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.metric_net.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.metric_net], ["metric_net"], dir=self.name, device=self.device
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        loss_dict = {
            f"{self.name}/loss/regression_loss": loss.item(),
            f"{self.name}/lr/W_lr": self.lr_scheduler.get_last_lr()[0],
            f"{self.name}/cvstem/alpha": self.cvstem_info["alpha"],
            f"{self.name}/cvstem/chi": self.cvstem_info["chi"],
            f"{self.name}/cvstem/nu": self.cvstem_info["nu"],
            f"{self.name}/cvstem/J": self.cvstem_info["J"],
        }
        loss_dict.update(grad_dict)

        self.eval()
        update_time = time.time() - t0
        self.num_updates += 1
        return loss_dict, {}, update_time

    def _plot_cvstem_diagnostics(self, M: torch.Tensor):
        import matplotlib.pyplot as plt

        with torch.no_grad():
            eig = torch.linalg.eigvalsh(M).cpu().numpy()
        cond = eig[:, -1] / np.clip(eig[:, 0], 1e-8, None)
        info = self.cvstem_info

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(eig.ravel(), bins=40)
        ax[0].set_title("Eigenvalues of optimal M (CV-STEM)")
        ax[0].set_xlabel("eigenvalue")
        ax[1].hist(cond, bins=40)
        ax[1].set_title(f"cond(M) per sample (chi={info['chi']:.3g})")
        ax[1].set_xlabel("condition number")
        for a in ax:
            a.grid(True, ls="--", alpha=0.5)
        fig.suptitle(
            f"CV-STEM (alpha={info['alpha']:.3g}, nu={info['nu']:.3g}, "
            f"status={info['status']})"
        )
        plt.tight_layout()
        plt.close(fig)
        return fig
