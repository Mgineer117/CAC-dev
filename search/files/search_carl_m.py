"""WandB sweep launcher for CARL_M."""

from sweep_common import LR_LOGUNIFORM, apply_cmg_rl_config, launch_sweep

PARAMETERS = {
    "lbd": {"min": 0.01, "max": 1.0},
    "actor_lr": LR_LOGUNIFORM,
    "critic_lr": LR_LOGUNIFORM,
    "W_lr": LR_LOGUNIFORM,
    "policy_updates_per_cmg_update": {"values": [1, 5, 10, 30]},
}

if __name__ == "__main__":
    launch_sweep(
        algo_name="carl_m",
        parameters=PARAMETERS,
        apply_config=apply_cmg_rl_config,
        default_project="CARL-M-SWEEP",
        default_count=4,
    )
