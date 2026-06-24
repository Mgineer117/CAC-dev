"""Shared WandB-sweep scaffolding for every ``search_*.py`` launcher.

All launchers do the same thing: build a sweep config, spin up an agent, and for
each trial pull ``wandb.config`` into ``args`` and call ``main.run``. Only the
swept parameters and the per-trial config-to-args mapping differ, so those are
the two things each launcher supplies; everything else lives here.

This file sits in ``<repo>/search/files/`` and the launchers are run from the
repo root (the configs in ``get_args`` use CWD-relative paths). We insert the
repo root onto ``sys.path`` so ``from main import run`` resolves either way.
"""

import argparse
import datetime
import os
import random
import sys
import uuid

# Make the repo root importable regardless of where the launcher is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import wandb  # noqa: E402

from main import run  # noqa: E402
from utils.get_args import get_args  # noqa: E402

# Shared CMG / contraction sweep dimensions reused across the CMG-RL launchers.
LR_LOGUNIFORM = {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"}
W_BOUND_LB = {"min": 0.01, "max": 1.0, "distribution": "log_uniform_values"}
W_BOUND_UB = {"min": 10.0, "max": 1000.0, "distribution": "log_uniform_values"}


def apply_cmg_rl_config(args, config):
    """Copy the CMG-RL family's swept hyperparameters (carl / carl_m / corl).

    Every key is optional: only values present in this trial's ``config`` are
    applied, so the same function serves launchers with different sweep spaces.
    """
    for key in (
        "lbd",
        "w_lb",
        "w_ub",
        "actor_lr",
        "critic_lr",
        "W_lr",
        "control_scaler",
        "policy_updates_per_cmg_update",
    ):
        if key in config:
            setattr(args, key, config[key])


def launch_sweep(
    *,
    algo_name,
    parameters,
    apply_config,
    default_project,
    default_count=5,
    metric="eval/performance_score",
    add_cli=None,
    extra_argv=None,
    prep_args=None,
):
    """Create or join a WandB sweep and run an agent.

    Args:
        algo_name: Value forced onto ``args.algo_name`` (and the ``--algo-name``
            CLI arg) for every trial.
        parameters: The sweep ``parameters`` dict, or a callable
            ``search_args -> dict`` when the space depends on launcher flags.
        apply_config: ``(args, config) -> None``; copies swept values onto args.
        default_project: Default ``--project`` value.
        default_count: Default number of trials this agent runs.
        metric: Sweep objective logged by training (maximized).
        add_cli: Optional ``parser -> None`` hook to register extra launcher
            flags (e.g. ``--task`` / ``--policy``).
        extra_argv: Optional ``search_args -> list[str]`` of args appended to the
            forwarded ``sys.argv`` (e.g. ``["--task", task]``).
        prep_args: Optional ``(args, search_args) -> None`` hook applied after the
            standard per-trial setup (fixed configs, schedule shortening, ...).
    """
    parser = argparse.ArgumentParser(description=f"WandB Sweep Launcher for {algo_name.upper()}")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID to join.")
    parser.add_argument("--count", type=int, default=default_count, help="Trials this agent runs.")
    parser.add_argument("--project", type=str, default=default_project, help="WandB project name.")
    if add_cli is not None:
        add_cli(parser)

    search_args, remaining_args = parser.parse_known_args()

    # Forward unknown args to get_args(); pin the algorithm and any launcher extras.
    forwarded = remaining_args + ["--algo-name", algo_name]
    if extra_argv is not None:
        forwarded += extra_argv(search_args)
    sys.argv = [sys.argv[0]] + forwarded

    torch.set_default_dtype(torch.float32)

    def train():
        # wandb.agent calls wandb.init() for us, but doing it explicitly exposes
        # wandb.config in this process.
        wandb.init()
        args = get_args()
        apply_config(args, wandb.config)

        args.seed = random.randint(1, 10000)
        args.num_runs = 1
        args.algo_name = algo_name
        if prep_args is not None:
            prep_args(args, search_args)

        unique_id = str(uuid.uuid4())[:4]
        exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
        print("-------------------------------------------------------")
        print(f"      {algo_name.upper()} Sweep Trial ID: {unique_id}")
        print(f"      Seed: {args.seed}")
        print(f"      Time Begun: {exp_time}")
        print("-------------------------------------------------------")

        run(args, args.seed, unique_id, exp_time)

    resolved_params = parameters(search_args) if callable(parameters) else parameters

    if search_args.sweep_id is None:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": metric, "goal": "maximize"},
            "parameters": resolved_params,
        }
        sweep_id = wandb.sweep(sweep_config, project=search_args.project)
        print("\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print("To run additional agents in parallel, run with --sweep_id")
        print(f"  --sweep_id {sweep_id} --project {search_args.project}")
        print("=======================================================\n")
        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")

    print(f"Starting wandb agent for sweep {sweep_id}")
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
