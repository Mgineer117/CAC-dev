"""WandB sweep launcher for C3M.

Searches the contraction rate ``lbd``, the CMG regularization ``eps``, the metric
LR ``W_lr``, the controller LR (``actor_lr`` -> ``u_lr``), the actor architecture,
and the number of CMG updates per controller update. The CMG architecture stays
pinned, and each trial is shortened to 1/5 of the configured training length so
the search covers more of the space per wall-clock hour.
"""

from sweep_common import LR_LOGUNIFORM, launch_sweep

PARAMETERS = {
    "lbd": {"min": 0.01, "max": 3.0},
    "eps": {"min": 1e-3, "max": 1.0, "distribution": "log_uniform_values"},
    "W_lr": LR_LOGUNIFORM,
    "actor_lr": LR_LOGUNIFORM,
    "actor_architecture": {"values": ["RL", "CL"]},
    "cmg_updates_per_policy_update": {"values": [1, 5, 10, 30]},
}


def apply_config(args, config):
    if "lbd" in config:
        args.lbd = config["lbd"]
    if "eps" in config:
        args.eps = config["eps"]
    if "W_lr" in config:
        args.W_lr = config["W_lr"]
    # C3M's "actor" is the deterministic controller; its LR is u_lr.
    if "actor_lr" in config:
        args.u_lr = config["actor_lr"]
    # Routed onto policy_type, the actor selector consumed by get_policy().
    if "actor_architecture" in config:
        args.policy_type = config["actor_architecture"]
    if "cmg_updates_per_policy_update" in config:
        args.cmg_updates_per_policy_update = config["cmg_updates_per_policy_update"]


def prep_args(args, search_args):
    # Fixed CMG architecture (not searched). u_lr / W_lr are now swept, and
    # w_lb / w_ub keep their get_args defaults (no override here).
    args.cmg_activation = "tanh"
    args.cmg_hidden_dims = [256, 256]

    # Shorten each trial to 1/5 of the configured C3M training length.
    if getattr(args, "epochs", None) is not None:
        args.epochs = max(1, int(args.epochs / 2))


if __name__ == "__main__":
    launch_sweep(
        algo_name="c3m",
        parameters=PARAMETERS,
        apply_config=apply_config,
        default_project="C3M-SWEEP",
        default_count=20,
        prep_args=prep_args,
    )
