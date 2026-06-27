"""Generic PPO sweep launcher (no dedicated run_*.bash; kept for ad-hoc use)."""

from sweep_common import launch_sweep

PARAMETERS = {
    "target_kl": {"min": 0.003, "max": 0.03},
    "learning_rate": {"min": 1e-5, "max": 1e-3},
    "gae": {"min": 0.8, "max": 1.0},
    "entropy_scaler": {"min": 1e-4, "max": 1e-1},
}


def apply_config(args, config):
    if "target_kl" in config:
        args.target_kl = config.target_kl
    if "learning_rate" in config:
        args.actor_lr = config.learning_rate
        args.critic_lr = config.learning_rate
    if "gae" in config:
        args.gae = config.gae
    if "entropy_scaler" in config:
        args.entropy_scaler = config.entropy_scaler


def prep_args(args, search_args):
    if getattr(args, "timesteps", None) is not None:
        args.timesteps = max(1, int(args.timesteps / 2))


if __name__ == "__main__":
    launch_sweep(
        algo_name="ppo",
        parameters=PARAMETERS,
        apply_config=apply_config,
        default_project="AEOS-SWEEP",
        default_count=20,
        metric="max_eval_return",
        prep_args=prep_args,
    )
