import datetime
import random
import uuid
import wandb
import torch

from utils.get_args import get_args
from main import run

def train():
    # wandb.agent automatically calls wandb.init() behind the scenes,
    # but calling it explicitly allows us to access wandb.config locally.
    wandb.init()
    config = wandb.config
    
    # Get standard args
    args = get_args()
    
    # Override args with sweep config
    if "lbd" in config:
        args.lbd = config.lbd
    if "w_lb" in config:
        args.w_lb = config.w_lb
    if "w_ub" in config:
        args.w_ub = config.w_ub
    if "u_lr" in config:
        args.u_lr = config.u_lr
    if "W_lr" in config:
        args.W_lr = config.W_lr
    if "cmg_hidden_dims" in config:
        args.cmg_hidden_dims = config.cmg_hidden_dims
    if "cmg_activation" in config:
        args.cmg_activation = config.cmg_activation
        
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
    
    print(f"-------------------------------------------------------")
    print(f"      C3M CMG Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"      CMG activation: {args.cmg_activation}")
    print(f"      CMG dims: {args.cmg_hidden_dims}")
    print(f"-------------------------------------------------------")
    
    # Run training
    if hasattr(args, "run_id"):
        run(args, seed, unique_id, exp_time, args.run_id)
    else:
        run(args, seed, unique_id, exp_time)

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher for C3M with CMG architecture variation")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID to join")
    parser.add_argument("--count", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--project", type=str, default="C3M-CMG-SWEEP", help="WandB project name")
    
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
                    "max": 1.0
                },
                "w_lb": {
                    "min": 0.01,
                    "max": 1.0,
                    "distribution": "log_uniform_values"
                },
                "w_ub": {
                    "min": 10.0,
                    "max": 1000.0,
                    "distribution": "log_uniform_values"
                },
                "u_lr": {
                    "min": 1e-5,
                    "max": 1e-3,
                    "distribution": "log_uniform_values"
                },
                "W_lr": {
                    "min": 1e-5,
                    "max": 1e-3,
                    "distribution": "log_uniform_values"
                },
                "cmg_activation": {
                    "values": ["siren", "tanh", "relu"]
                },
                "cmg_hidden_dims": {
                    "values": [
                        [64, 64],
                        [128, 128],
                        [256, 256],
                        [128, 128, 128],
                        [256, 256, 256]
                    ]
                }
            }
        }
        
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=search_args.project) 
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search_c3m_cmg.py --sweep_id {sweep_id}")
        print(f"=======================================================\n")
        
        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")
    
    print(f"Starting wandb agent for sweep {sweep_id}")
    # Count controls how many trials this specific agent will run
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
