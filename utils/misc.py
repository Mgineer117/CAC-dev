import json
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from log.wandb_logger import WandbLogger


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def temp_seed(seed, pid):
    """
    This saves current seed info and calls after stochastic action selection.
    -------------------------------------------------------------------------
    This is to introduce the stochacity in each multiprocessor.
    Without this, the samples from each multiprocessor will be same since the seed was fixed
    """
    rand_int = random.randint(0, 1_000_000)  # create a random integer
    seed = seed + pid + rand_int

    # Set the temporary seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed


def setup_logger(args, unique_id, exp_time, seed, verbose=True):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join(
            (args.algo_name, args.task, unique_id, "seed:" + str(seed))
        )

    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)
    args.unique_id = unique_id

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
    )
    logger.save_config(default_cfg, verbose=verbose)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer


def override_args(init_args):
    """
    Loads task and algorithm configurations and backfills missing (None) arguments.
    """
    # Create a copy to avoid side effects on the input object
    args = deepcopy(init_args)

    # Load configurations
    env_params = load_hyperparams(file_path=f"config/task/{args.task}.json")
    algo_params = load_hyperparams(file_path=f"config/algorithm/{args.algo_name}.json")

    # Combine parameters (Algorithm defaults usually override Task defaults if keys overlap)
    # Using {**x, **y} creates a new dict merging both; keys in y overwrite x.
    combined_defaults = {**env_params, **algo_params}

    # Apply defaults if the argument is None or doesn't exist
    for k, v in combined_defaults.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # lbd is per-algo per-task: read "{task}_lbd" from the algo JSON if present.
    task_lbd_key = f"{args.task}_lbd"
    if task_lbd_key in algo_params:
        args.lbd = algo_params[task_lbd_key]

    return args


# --- Architecture sweep helpers (shared by the search_*.py launchers) --------- #
# Uniform-width hidden layers: hidden_dims = [width] * depth.
ARCH_WIDTHS = [64, 128, 256, 512, 1024]
ARCH_DEPTHS = [1, 2, 3, 4]


def arch_sweep_parameters(include_cmg: bool = True, include_actor: bool = True) -> dict:
    """Returns wandb-sweep parameter entries for CMG / actor architecture search.

    Width and depth are swept separately (not as a giant list-of-lists) so the
    optimizer sees clean categorical dimensions; hidden_dims = [width] * depth is
    reconstructed in apply_arch_config().
    """
    params = {}
    if include_cmg:
        params["cmg_width"] = {"values": ARCH_WIDTHS}
        params["cmg_depth"] = {"values": ARCH_DEPTHS}
        params["cmg_activation"] = {"values": ["tanh", "relu"]}
    if include_actor:
        params["actor_width"] = {"values": ARCH_WIDTHS}
        params["actor_depth"] = {"values": ARCH_DEPTHS}
        # NOTE: "siren" is a network type, not a pointwise activation, and is only
        # wired into the C3M CLActor (get_u_model). Sweeps over RLActor-based algos
        # (CARL/CORL) must not include it. C3M sweeps override this list explicitly.
        params["actor_activation"] = {"values": ["tanh", "relu", "elu"]}
    return params


def apply_arch_config(args, config) -> None:
    """Applies CMG / actor architecture sweep values from a wandb config to args.

    Accepts either the (width, depth) form produced by arch_sweep_parameters() or
    an explicit hidden-dims list (cmg_hidden_dims / actor_dim).
    """
    # CMG
    if "cmg_width" in config and "cmg_depth" in config:
        args.cmg_hidden_dims = [int(config.cmg_width)] * int(config.cmg_depth)
    elif "cmg_hidden_dims" in config:
        args.cmg_hidden_dims = list(config.cmg_hidden_dims)
    if "cmg_activation" in config:
        args.cmg_activation = config.cmg_activation

    # Actor
    if "actor_width" in config and "actor_depth" in config:
        args.actor_dim = [int(config.actor_width)] * int(config.actor_depth)
    elif "actor_dim" in config:
        args.actor_dim = list(config.actor_dim)
    if "actor_activation" in config:
        args.actor_activation = config.actor_activation


def load_hyperparams(file_path):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams  # .get({})
    except FileNotFoundError:
        print(f"No file found at {file_path}. Returning default empty dictionary.")
        return {}


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")
