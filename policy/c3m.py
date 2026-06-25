import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class C3M(Base):
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
        W_tol: float = 1e-5,
        W_patience: int = 20,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super(C3M, self).__init__()

        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates

        self.CMG = CMG
        self.actor = actor

        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            self.get_f_and_B.eval()

        self.eps = eps
        self.W_tol = W_tol
        self.W_patience = W_patience
        self.w_ub = w_ub
        self.w_lb = w_lb
        self.lbd = lbd

        self.W_patience_counter = 0
        self.best_W_loss = float("inf")
        self.stop_W_training = False

        self.W_optimizer = torch.optim.Adam(self.CMG.parameters(), lr=W_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=u_lr)

        self.W_lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.lr_decay_lambda)
        self.actor_lr_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.lr_decay_lambda)

        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def forward(self, state: np.ndarray):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        a, _ = self.actor(state)

        return a, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def compute_loss(self):
        I = torch.eye(self.x_dim, device=self.device)

        # === SAMPLE BATCH === #
        buffer_size, batch_size = self.data["x"].shape[0], 1024
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        batch = {key: self.data[key][indices] for key in self.data.keys()}

        # === PREPARE TENSORS === #
        x = self.to_tensor(batch["x"]).requires_grad_()
        xref = self.to_tensor(batch["xref"])
        uref = self.to_tensor(batch["uref"])

        raw_W, _ = self.CMG(x)
        W = self._bound_W(raw_W)
        M = torch.linalg.solve(W, I.unsqueeze(0).expand(W.shape[0], -1, -1))

        f, B, _ = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)
        B = B.to(self._dtype).to(self.device)

        DfDx = self.Jacobian(f, x)
        DBDx = self.B_Jacobian(B, x)

        f = f.detach()
        B = B.detach()

        state = torch.concatenate([x, xref, uref], dim=1)
        u, _ = self.actor(state)
        K = self.Jacobian(u, x)

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.u_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M

        overshoot = W - self.w_ub * I

        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)

        pd_loss, pd_reg = self.loss_pos_matrix_random_sampling(-Cu)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_random_sampling(-overshoot)

        self.record_eigenvalues(Cu, dot_M, sym_MABK, overshoot)

        loss = overshoot_loss + pd_loss + pd_reg + overshoot_reg

        return loss, {"pd_loss": pd_loss, "overshoot_loss": overshoot_loss}

    def optimize_params(self, loss: torch.Tensor):
        self.W_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()

        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in self.parameters()
        ):
            self.W_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            return {}

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.CMG, self.actor, self.lbd],
            ["CMG", "actor", "lbd"],
            dir="C3M",
            device=self.device,
        )
        self.W_optimizer.step()
        self.actor_optimizer.step()
        
        if getattr(self, "W_lr_scheduler", None) is not None:
            self.W_lr_scheduler.step()
        if getattr(self, "actor_lr_scheduler", None) is not None:
            self.actor_lr_scheduler.step()

        return grad_dict

    def learn(self):
        self.train()
        t0 = time.time()

        self.progress = min(1.0, self.num_updates / max(1, self.nupdates))

        loss, infos = self.compute_loss()
        grad_dict = self.optimize_params(loss)

        supp_dict = {}
        if self.num_updates % 500 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["C3M/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/lr/W_lr": self.W_lr_scheduler.get_last_lr()[0] if hasattr(self, "W_lr_scheduler") else 3e-4,
            f"{self.name}/lr/u_lr": self.actor_lr_scheduler.get_last_lr()[0] if hasattr(self, "actor_lr_scheduler") else 1e-4,
        }
        loss_dict.update(grad_dict)

        self.eval()
        update_time = time.time() - t0
        self.num_updates += 1

        return loss_dict, supp_dict, update_time
