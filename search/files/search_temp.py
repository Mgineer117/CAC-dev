"""WandB sweep launcher for TEMP.

The optimal policy (SAC or PPO) is fixed per sweep via ``--policy`` so the search
never mixes the two incompatible parameter spaces; SAC-only knobs are simply
omitted from PPO sweeps.
"""

from sweep_common import LR_LOGUNIFORM, launch_sweep

# Dimensions shared by both optimal-policy types.
SHARED_PARAMS = {
    "temp_gamma_contracting": {"values": [0.0, 0.1]},
    "temp_gamma_optimal": {"values": [0.3, 0.6, 0.9]},
    "temp_cmg_updates_per_iter": {"values": [1, 2, 5, 10]},
    "critic_hidden_size": {"values": [128, 256, 512]},
    "critic_depth": {"values": [2, 3, 4]},
    "actor_activation": {"values": ["tanh", "relu", "elu"]},
    "actor_lr": LR_LOGUNIFORM,
    "W_lr": LR_LOGUNIFORM,
    "lbd": {"min": 0.01, "max": 3.0},
}

# Extra dimensions only meaningful when the optimal policy is SAC.
SAC_PARAMS = {
    "sac_tau": {"min": 1e-3, "max": 5e-2, "distribution": "log_uniform_values"},
    "sac_alpha_lr": LR_LOGUNIFORM,
    "sac_init_alpha": {"min": 0.01, "max": 1.0, "distribution": "log_uniform_values"},
    "sac_batch_size": {"values": [64, 512]},
    "sac_utd": {"values": [1, 8]},
    "sac_learning_starts": {"values": [100, 10000]},
    "critic_lr": LR_LOGUNIFORM,
}


def build_parameters(search_args):
    params = dict(SHARED_PARAMS)
    if search_args.policy == "sac":
        params.update(SAC_PARAMS)
    # Pin the optimal policy so the sweep stays in one parameter space.
    params["temp_optimal_policy"] = {"value": search_args.policy}
    return params


def apply_config(args, config):
    # CMG / contraction + shared LRs.
    for key in ("lbd", "w_lb", "w_ub", "W_lr", "actor_lr", "critic_lr"):
        if key in config:
            setattr(args, key, config[key])

    # SAC knobs.
    for key in (
        "sac_tau",
        "sac_alpha_lr",
        "sac_init_alpha",
        "sac_batch_size",
        "sac_utd",
        "sac_learning_starts",
    ):
        if key in config:
            setattr(args, key, config[key])

    # TEMP-specific.
    for key in (
        "temp_optimal_policy",
        "temp_gamma_contracting",
        "temp_gamma_optimal",
        "temp_cmg_updates_per_iter",
        "actor_activation",
    ):
        if key in config:
            setattr(args, key, config[key])

    # Critic architecture (uniform-width hidden layers).
    if "critic_hidden_size" in config or "critic_depth" in config:
        h = config.get("critic_hidden_size", 256)
        d = config.get("critic_depth", 2)
        args.critic_dim = [h] * d


def add_cli(parser):
    parser.add_argument("--task", type=str, default="cartpole")
    parser.add_argument(
        "--policy", type=str, default="sac", choices=["sac", "ppo"],
        help="Fix the optimal policy type for this sweep (sac or ppo).",
    )


if __name__ == "__main__":
    launch_sweep(
        algo_name="temp",
        parameters=build_parameters,
        apply_config=apply_config,
        default_project="TEMP-SWEEP",
        default_count=100,
        add_cli=add_cli,
        extra_argv=lambda sa: ["--task", sa.task],
    )
