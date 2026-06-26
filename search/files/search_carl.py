"""WandB sweep launcher for CARL."""

from sweep_common import (
    LR_LOGUNIFORM,
    apply_cmg_rl_config,
    launch_sweep,
)

PARAMETERS = {
    "lbd": {"min": 0.01, "max": 1.0},
    "w_lb": {"min": 0.01, "max": 1.0, "distribution": "log_uniform_values"},
    "w_ub": {"min": 10.0, "max": 1000.0, "distribution": "log_uniform_values"},
    "actor_lr": LR_LOGUNIFORM,
    "critic_lr": LR_LOGUNIFORM,
    "W_lr": LR_LOGUNIFORM,
    "policy_updates_per_cmg_update": {"values": [1, 5, 10, 20, 50, 100]},
    "control_scaler": {"min": 0.0, "max": 1.0},
}

if __name__ == "__main__":
    launch_sweep(
        algo_name="carl",
        parameters=PARAMETERS,
        apply_config=apply_cmg_rl_config,
        default_project="CARL-SWEEP",
        default_count=20,
    )
