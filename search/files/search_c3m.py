"""WandB sweep launcher for C3M.

Only the contraction rate ``lbd`` is searched; the CMG architecture and learning
rates are pinned, and each trial is shortened to 1/5 of the configured training
length so the search covers more of the space per wall-clock hour.
"""

from sweep_common import launch_sweep

PARAMETERS = {
    "lbd": {"min": 0.01, "max": 3.0},
}


def apply_config(args, config):
    if "lbd" in config:
        args.lbd = config.lbd


def prep_args(args, search_args):
    # Fixed CMG configuration (no longer searched).
    args.cmg_activation = "tanh"
    args.cmg_hidden_dims = [256, 256]
    args.u_lr = 1e-4
    args.W_lr = 3e-4
    args.w_lb = 0.05
    args.w_ub = 100.0

    # Shorten each trial to 1/5 of the configured C3M training length.
    if getattr(args, "epochs", None) is not None:
        args.epochs = max(1, int(args.epochs / 5))


if __name__ == "__main__":
    launch_sweep(
        algo_name="c3m",
        parameters=PARAMETERS,
        apply_config=apply_config,
        default_project="C3M-SWEEP",
        default_count=5,
        prep_args=prep_args,
    )
