import datetime
import random
import uuid
import wandb
import torch

from utils.get_args import get_args
from main import run


def train():
    wandb.init()
    config = wandb.config

    args = get_args()

    if "lbd" in config:
        args.lbd = config.lbd
    if "w_lb" in config:
        args.w_lb = config.w_lb
    if "w_ub" in config:
        args.w_ub = config.w_ub
    if "actor_lr" in config:
        args.actor_lr = config.actor_lr
    if "critic_lr" in config:
        args.critic_lr = config.critic_lr
    if "W_lr" in config:
        args.W_lr = config.W_lr
    if "control_scaler" in config:
        args.control_scaler = config.control_scaler
    if "policy_updates_per_cmg_update" in config:
        args.policy_updates_per_cmg_update = config.policy_updates_per_cmg_update
    if "c3m_pretrain_cmg" in config:
        args.c3m_pretrain_cmg = config.c3m_pretrain_cmg
    if "c3m_pretrain_c1c2" in config:
        args.c3m_pretrain_c1c2 = config.c3m_pretrain_c1c2

    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    seed = random.randint(1, 10000)
    args.seed = seed
    args.num_runs = 1
    args.algo_name = "carl_m"

    print(f"-------------------------------------------------------")
    print(f"      CARL_M Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"-------------------------------------------------------")

    if hasattr(args, "run_id"):
        run(args, seed, unique_id, exp_time, args.run_id)
    else:
        run(args, seed, unique_id, exp_time)


import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher for CARL_M")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID to join")
    parser.add_argument("--count", type=int, default=4, help="Number of trials to run")
    parser.add_argument("--project", type=str, default="CARL-M-SWEEP", help="WandB project name")

    search_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args + ["--algo-name", "carl_m"]

    torch.set_default_dtype(torch.float32)

    if search_args.sweep_id is None:
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "eval/performance_score",
                "goal": "maximize"
            },
            "parameters": {
                "lbd": {
                    "min": 0.01,
                    "max": 1.0
                },
                "actor_lr": {
                    "min": 1e-5,
                    "max": 1e-3,
                    "distribution": "log_uniform_values"
                },
                "critic_lr": {
                    "min": 1e-5,
                    "max": 1e-3,
                    "distribution": "log_uniform_values"
                },
                "W_lr": {
                    "min": 1e-5,
                    "max": 1e-3,
                    "distribution": "log_uniform_values"
                },
                "policy_updates_per_cmg_update": {
                    "values": [1, 5, 10, 30]
                },
                "c3m_pretrain_cmg": {
                    "values": [True, False]
                },
                "c3m_pretrain_c1c2": {
                    "values": [True, False]
                },
            }
        }

        sweep_id = wandb.sweep(sweep_config, project=search_args.project)
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search_carl_m.py --sweep_id {sweep_id}")
        print(f"=======================================================\n")

        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")

    print(f"Starting wandb agent for sweep {sweep_id}")
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
