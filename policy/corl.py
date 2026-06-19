from typing import Callable

import torch
import torch.nn as nn
from torch import inverse, transpose

from policy.carl import CARL
from policy.cmg_pretrain import SDLQRPretrainMixin


class CORL(SDLQRPretrainMixin, CARL):
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
         This recipe lives in :class:`SDLQRPretrainMixin`.
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
        pretrain_epochs: int = 5000,
        pretrain_buffer_size: int = 10000,
        pretrain_minibatch_size: int = 1024,
        pretrain_log_interval: int = 100,
        pretrain_W_lr: float = 1e-3,
        # early stopping (validation total-loss + plateau-slope + restore-best)
        val_split: float = 0.1,
        val_batch_size: int = 512,
        val_interval: int = 25,
        plateau_window: int = 5,
        plateau_tol: float = 1e-3,
        plateau_patience: int = 3,
        restore_best: bool = True,
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

        # logging handles (optional; set by get_policy). Pretraining happens in
        # __init__ before the trainer exists, so we log here directly.
        self.logger = logger
        self.writer = writer

        # CORL pretrains the CMG with its dedicated CMG optimizer and freezes it after.
        self.cmg_pretrain_optimizer = self.W_optimizer
        self.freeze_cmg_after_pretrain = True

        # running-average reward normalizer (EMA of |raw reward| magnitude)
        self.reward_norm_beta = 0.99
        self.running_reward_scale = None

        # Pretrain the contraction metric using SD-LQR controls.
        self.pretrain_CMG()

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
