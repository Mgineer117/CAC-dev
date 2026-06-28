import argparse
import datetime
import random
import sys
import uuid

import torch
import wandb

from main import run
from utils.get_args import get_args


def train():
    wandb.init()
    config = wandb.config

    args = get_args()

    # --- CMG / contraction ---
    if "lbd" in config:
        args.lbd = config.lbd
    if "w_lb" in config:
        args.w_lb = config.w_lb
    if "w_ub" in config:
        args.w_ub = config.w_ub
    if "W_lr" in config:
        args.W_lr = config.W_lr

    # --- shared RL network LRs ---
    if "actor_lr" in config:
        args.actor_lr = config.actor_lr
    if "critic_lr" in config:
        args.critic_lr = config.critic_lr

    # --- SAC hyperparameters ---
    if "sac_tau" in config:
        args.sac_tau = config.sac_tau
    if "sac_alpha_lr" in config:
        args.sac_alpha_lr = config.sac_alpha_lr
    if "sac_init_alpha" in config:
        args.sac_init_alpha = config.sac_init_alpha
    if "sac_batch_size" in config:
        args.sac_batch_size = config.sac_batch_size
    if "sac_utd" in config:
        args.sac_utd = config.sac_utd
    if "sac_learning_starts" in config:
        args.sac_learning_starts = config.sac_learning_starts

    # --- TEMP-specific ---
    if "temp_optimal_policy" in config:
        args.temp_optimal_policy = config.temp_optimal_policy
    if "temp_gamma_contracting" in config:
        args.temp_gamma_contracting = config.temp_gamma_contracting
    if "temp_gamma_optimal" in config:
        args.temp_gamma_optimal = config.temp_gamma_optimal
    if "temp_cmg_updates_per_iter" in config:
        args.temp_cmg_updates_per_iter = config.temp_cmg_updates_per_iter
    if "critic_hidden_size" in config or "critic_depth" in config:
        h = config.get("critic_hidden_size", 256)
        d = config.get("critic_depth", 2)
        args.critic_dim = [h] * d
    if "actor_activation" in config:
        args.actor_activation = config.actor_activation

    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    seed = random.randint(1, 10000)
    args.seed = seed
    args.num_runs = 1
    args.algo_name = "temp"

    print(f"-------------------------------------------------------")
    print(f"      TEMP Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"-------------------------------------------------------")

    if hasattr(args, "run_id"):
        run(args, seed, unique_id, exp_time, args.run_id)
    else:
        run(args, seed, unique_id, exp_time)


# Parameters shared by both SAC and PPO optimal policy.
SHARED_PARAMS = {
    # --- discount factors ---
    "temp_gamma_contracting": {
        "values": [0.0, 0.1]
    },
    "temp_gamma_optimal": {
        "values": [0.3, 0.6, 0.9]
    },
    # --- CMG update frequency ---
    "temp_cmg_updates_per_iter": {
        "values": [1, 2, 5, 10]
    },
    # --- critic architecture ---
    "critic_hidden_size": {
        "values": [128, 256, 512]
    },
    "critic_depth": {
        "values": [2, 3, 4]
    },
    # --- activation function ---
    "actor_activation": {
        "values": ["tanh", "relu", "elu"]
    },
    # --- shared network LR ---
    "actor_lr": {
        "min": 1e-5,
        "max": 1e-3,
        "distribution": "log_uniform_values",
    },
    # --- CMG / contraction ---
    "W_lr": {
        "min": 1e-5,
        "max": 1e-3,
        "distribution": "log_uniform_values",
    },
    "lbd": {
        "min": 0.01,
        "max": 3.0,
    },
}

# Extra parameters only meaningful when using SAC as the optimal policy.
SAC_PARAMS = {
    "sac_tau": {
        "min": 1e-3,
        "max": 5e-2,
        "distribution": "log_uniform_values",
    },
    "sac_alpha_lr": {
        "min": 1e-5,
        "max": 1e-3,
        "distribution": "log_uniform_values",
    },
    "sac_init_alpha": {
        "min": 0.01,
        "max": 1.0,
        "distribution": "log_uniform_values",
    },
    "sac_batch_size": {
        "values": [64, 512]
    },
    "sac_utd": {
        "values": [1, 8]
    },
    "sac_learning_starts": {
        "values": [100, 10000]
    },
    "critic_lr": {
        "min": 1e-5,
        "max": 1e-3,
        "distribution": "log_uniform_values",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher for TEMP")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--project", type=str, default="TEMP-SWEEP")
    parser.add_argument("--task", type=str, default="cartpole")
    parser.add_argument("--policy", type=str, default="sac", choices=["sac", "ppo"],
                        help="Fix the optimal policy type for this sweep (sac or ppo).")

    search_args, remaining_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_args + ["--algo-name", "temp", "--task", search_args.task]

    torch.set_default_dtype(torch.float32)

    if search_args.sweep_id is None:
        policy_params = {**SHARED_PARAMS, **SAC_PARAMS} if search_args.policy == "sac" else SHARED_PARAMS
        # Fix the optimal policy so the sweep doesn't mix incompatible param spaces.
        policy_params["temp_optimal_policy"] = {"value": search_args.policy}

        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "eval/performance_score",
                "goal": "maximize",
            },
            "parameters": policy_params,
        }

        sweep_id = wandb.sweep(sweep_config, project=search_args.project)
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search_temp.py --sweep_id {sweep_id} --task {search_args.task} --policy {search_args.policy}")
        print(f"=======================================================\n")

        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")

    print(f"Starting wandb agent for sweep {sweep_id}")
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
