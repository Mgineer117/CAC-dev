import datetime
import random
import uuid
import wandb
import torch

from utils.get_args import get_args
from utils.misc import apply_arch_config, arch_sweep_parameters
from main import run

def train():
    # wandb.agent automatically calls wandb.init() behind the scenes,
    # but calling it explicitly allows us to access wandb.config locally.
    wandb.init()
    config = wandb.config

    # Get standard args
    args = get_args()

    # --- searched hyperparameters ---
    if "lbd" in config:
        args.lbd = config.lbd
    # Whether to warm-start the CMG with the SD-LQR contraction (pd) pretraining.
    if "c3m_pretrain_cmg" in config:
        args.c3m_pretrain_cmg = bool(config.c3m_pretrain_cmg)
    # Whether to prepend a C1/C2 init phase BEFORE the contraction (pd) pretraining.
    # Pipeline when both are True:
    #   Phase 1: minimize C1/C2 loss (initialize network geometry)
    #   Phase 2: only enforce contraction (pd) loss
    # c3m_pretrain_c1c2 is forced to False when c3m_pretrain_cmg is False.
    if "c3m_pretrain_c1c2" in config:
        pretrain_cmg_active = getattr(args, "c3m_pretrain_cmg", False)
        args.c3m_pretrain_c1c2 = bool(config.c3m_pretrain_c1c2) and pretrain_cmg_active

    # --- fixed CMG configuration (no longer searched) ---
    # Bounded eigenvalue-sigmoid CMG with a SIREN backbone, depth-2 / width-256,
    # fixed contraction-metric bounds and learning rates.
    args.cmg_activation = "siren"
    args.cmg_hidden_dims = [256, 256]
    args.u_lr = 1e-4
    args.W_lr = 3e-4
    args.w_lb = 0.05
    args.w_ub = 100.0

    # Override actor architecture (width, depth, activation) from the sweep config.
    apply_arch_config(args, config)

    # Setup for the trial run
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    # Pick a random seed for this trial
    seed = random.randint(1, 10000)
    args.seed = seed

    # We set num_runs to 1 as we only want one execution per sweep combination
    args.num_runs = 1

    # Force algorithm to c3m
    args.algo_name = "c3m"

    # Shorten each sweep trial to 1/5 of the configured C3M training length so the
    # search explores more hyperparameter combinations in the same wall-clock budget.
    if getattr(args, "c3m_epochs", None) is not None:
        args.c3m_epochs = max(1, int(args.c3m_epochs / 5))

    print(f"-------------------------------------------------------")
    print(f"      C3M Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"-------------------------------------------------------")

    # Run training
    if hasattr(args, "run_id"):
        run(args, seed, unique_id, exp_time, args.run_id)
    else:
        run(args, seed, unique_id, exp_time)

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher for C3M")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID to join")
    parser.add_argument("--count", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--project", type=str, default="C3M-SWEEP", help="WandB project name")

    # Parse only known search-specific args
    search_args, remaining_args = parser.parse_known_args()

    # Overwrite sys.argv so get_args() in train() doesn't fail on unknown args
    sys.argv = [sys.argv[0]] + remaining_args + ["--algo-name", "c3m"]

    torch.set_default_dtype(torch.float32)

    if search_args.sweep_id is None:
        # Define the hyperparameter search space
        sweep_config = {
            "method": "bayes",  # Supports "bayes", "random", or "grid"
            "metric": {
                "name": "eval/performance_score",
                "goal": "maximize"
            },
            "parameters": {
                "lbd": {
                    "min": 0.01,
                    "max": 3.0
                },
                # Warm-start the CMG with SD-LQR contraction pretraining or not.
                "c3m_pretrain_cmg": {
                    "values": [True, False]
                },
                # Prepend a C1/C2 init phase before the contraction (pd) pretraining.
                # Pipeline when both c3m_pretrain_cmg=True and c3m_pretrain_c1c2=True:
                #   Phase 1: minimize C1/C2 loss (initialize network geometry)
                #   Phase 2: only enforce contraction (pd) loss
                # Forced to False in train() when c3m_pretrain_cmg=False.
                "c3m_pretrain_c1c2": {
                    "values": [True, False]
                },
                # Actor architecture only (width, depth, activation); the CMG
                # configuration is fixed in train(), so it is not searched.
                **arch_sweep_parameters(include_cmg=False, include_actor=True),
                # C3M's CLActor supports SIREN, so add it as an activation option
                # (overrides the actor_activation list from arch_sweep_parameters).
                "actor_activation": {"values": ["tanh", "relu", "elu", "siren"]},
            }
        }

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=search_args.project)
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search_c3m.py --sweep_id {sweep_id}")
        print(f"=======================================================\n")

        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")

    print(f"Starting wandb agent for sweep {sweep_id}")
    # Count controls how many trials this specific agent will run
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
