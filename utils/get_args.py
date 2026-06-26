import json
import os
import argparse
import torch


def get_args():
    p = argparse.ArgumentParser()

    # --- logging ---
    p.add_argument("--project", type=str, default="Exp", help="WandB project.")
    p.add_argument("--logdir", type=str, default="log/train_log", help="Logging folder.")
    p.add_argument("--group", type=str, default=None, help="Experiment group folder.")
    p.add_argument("--name", type=str, default=None, help="Seed-specific run name in group.")

    # --- experiment ---
    p.add_argument("--task", type=str, default="car", help="Task: [car, pvtol, neurallander, quadrotor].")
    p.add_argument("--algo-name", type=str, default="cac", help="Algorithm name.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-runs", type=int, default=10, help="Experiments per algorithm.")
    p.add_argument("--gpu-idx", type=int, default=0, help="GPU index.")
    p.add_argument("--rendering", action="store_true", help="Render environment as video to wandb.")
    p.add_argument("--load-pretrained-model", action="store_true")

    # --- policy ---
    p.add_argument("--policy-mode", type=str, default=None, help='["deterministic", "stochastic"].')
    p.add_argument("--policy-type", type=str, default=None, help='["CL", "RL"].')
    p.add_argument("--cmg-mode", type=str, default=None, help='CMG mode: ["deterministic", "stochastic"].')
    p.add_argument("--disable-cmg-training", action="store_true", help="Disable CMG training.")
    p.add_argument("--policy-updates-per-cmg-update", type=int, default=5)
    p.add_argument("--cmg-updates-per-policy-update", type=int, default=1, help="C3M: CMG updates per controller update.")

    # --- learning rates ---
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--dynamic-lr", type=float, default=1e-3, help="Dynamics approximator LR.")
    p.add_argument("--sdc-lr", type=float, default=1e-3, help="SDC decomposition net LR.")
    p.add_argument("--W-lr", type=float, default=None, help="CMG learning rate.")
    p.add_argument("--u-lr", type=float, default=None, help="C3M actor LR.")

    # --- contraction ---
    p.add_argument("--w-ub", type=float, default=10.0, help="Contraction metric upper bound.")
    p.add_argument("--w-lb", type=float, default=0.1, help="Contraction metric lower bound.")
    p.add_argument("--lbd", type=float, default=0.5, help="Desired contraction rate.")
    p.add_argument("--eps", type=float, default=0.1, help="CMG regularization.")

    # --- PPO / TRPO ---
    p.add_argument("--eps-clip", type=float, default=0.1, help="Epsilon clip for PPO.")
    p.add_argument("--k-epochs", type=int, default=5, help="PPO K epochs.")
    p.add_argument("--target-kl", type=float, default=0.003)
    p.add_argument("--gae", type=float, default=0.95, help="Generalized Advantage Estimation.")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")

    # --- NCM / CV-STEM ---
    p.add_argument("--cvstem-alpha", type=float, default=None, help="NCM: CV-STEM alpha (defaults to --lbd).")
    p.add_argument("--ncm-R-scaler", type=float, default=1.0, help="NCM: control weight R = R_scaler*I.")
    p.add_argument("--cvstem-dt", type=float, default=None, help="NCM: dt for (W-I)/dt term (defaults to env.dt).")
    p.add_argument("--cvstem-w-nu", type=float, default=1.0, help="NCM: weight on nu in CV-STEM objective.")
    p.add_argument("--cvstem-num-samples", type=int, default=100, help="NCM: states in joint CV-STEM SDP.")
    p.add_argument("--cvstem-no-linesearch", action="store_true", help="NCM: disable alpha line search.")
    p.add_argument("--cvstem-no-dwdt", action="store_true", help="NCM: drop (W-I)/dt term.")

    # --- SAC (and the SAC cores inside `temp`) ---
    p.add_argument("--sac-tau", type=float, default=5e-3, help="Polyak averaging coefficient for target critics.")
    p.add_argument("--sac-alpha-lr", type=float, default=3e-4, help="Entropy temperature learning rate.")
    p.add_argument("--sac-init-alpha", type=float, default=0.2, help="Initial entropy temperature alpha.")
    p.add_argument("--sac-no-autotune-alpha", action="store_true", help="Disable automatic entropy tuning (fix alpha).")
    p.add_argument("--sac-buffer-size", type=int, default=1_000_000, help="Replay buffer capacity.")
    p.add_argument("--sac-batch-size", type=int, default=256, help="Replay minibatch size for SAC updates.")
    p.add_argument("--sac-utd", type=float, default=1.0, help="Update-to-data ratio (gradient steps per new env step).")
    p.add_argument("--sac-learning-starts", type=int, default=5000, help="Env steps to collect before SAC updates begin.")

    # --- TEMP (contracting-policy CMG synthesis; replaces corl) ---
    p.add_argument("--temp-optimal-policy", type=str, default="sac", choices=["sac", "ppo"],
                   help="Deployed high-discount policy: a second SAC or on-policy PPO.")
    p.add_argument("--temp-gamma-contracting", type=float, default=0.0,
                   help="Discount of the contracting policy that drives the CMG (gamma -> 0).")
    p.add_argument("--temp-gamma-optimal", type=float, default=None,
                   help="Discount of the deployed optimal policy (defaults to --gamma).")
    p.add_argument("--temp-cmg-updates-per-iter", type=int, default=50,
                   help="CMG pd-loss gradient steps per training iteration.")

    # --- network architecture ---
    p.add_argument("--dynamic-dim", type=list, default=[256, 256], help="Dynamics net hidden dims.")
    p.add_argument("--cmg-hidden-dims", type=list, default=[128, 128])
    p.add_argument("--cmg-activation", type=str, default="tanh", help="['tanh', 'relu'].")
    p.add_argument("--sdc-dim", type=list, default=[256, 256], help="SDC net hidden dims.")
    p.add_argument("--actor-dim", type=list, default=[256, 256])
    p.add_argument("--actor-activation", type=str, default="tanh", help="['tanh', 'relu', 'elu', 'leaky_relu', 'gelu'].")
    p.add_argument("--critic-dim", type=list, default=[256, 256])

    # --- training schedule ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dynamics-epochs", type=int, default=20000)
    p.add_argument("--sdc-epochs", type=int, default=2000)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-num", type=int, default=10, help="Reference trajectories for evaluation.")

    # --- buffers & batches ---
    p.add_argument("--c3m-buffer-size", type=int, default=200_000)
    p.add_argument("--dynamics-buffer-size", type=int, default=100_000)
    p.add_argument("--sdc-buffer-size", type=int, default=100_000)
    p.add_argument("--num-minibatch", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=1024)

    # --- rewards & sampling ---
    p.add_argument("--sigma", type=float, default=0.0, help="Disturbance rate.")
    p.add_argument("--sample-mode", type=str, default="Uniform")
    p.add_argument("--reward-mode", type=str, default=None)
    p.add_argument("--anneal-stddev", action="store_true", default=None)
    p.add_argument("--entropy-scaler", type=float, default=0.0)
    p.add_argument("--W-entropy-scaler", type=float, default=None)
    p.add_argument("--control-scaler", type=float, default=None)

    args = p.parse_args()

    for config_path in [
        os.path.join("config", "task", f"{args.task}.json"),
        os.path.join("config", "algorithm", f"{args.algo_name}.json"),
    ]:
        if os.path.exists(config_path):
            with open(config_path) as f:
                for k, v in json.load(f).items():
                    if getattr(args, k, None) is None:
                        setattr(args, k, v)

    args.device = select_device(args.gpu_idx)
    return args


def select_device(device_arg=0, min_free_gb: float = 2.0, verbose=True):
    if device_arg == "cpu":
        device, name = torch.device("cpu"), "CPU"
    elif torch.cuda.is_available():
        idx = 0 if device_arg is None else int(device_arg)
        if idx >= torch.cuda.device_count():
            print(f"Warning: GPU {idx} requested but only {torch.cuda.device_count()} available. Falling back to CPU.")
            device, name = torch.device("cpu"), "CPU (fallback)"
        else:
            device = torch.device(f"cuda:{idx}")
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            free_gb, total_gb = free / 1e9, total / 1e9
            if free_gb < min_free_gb:
                print(
                    f"Warning: GPU {idx} has only {free_gb:.1f}GB free "
                    f"(need {min_free_gb}GB, likely too many concurrent processes). "
                    f"Falling back to CPU."
                )
                device, name = torch.device("cpu"), "CPU (low-VRAM fallback)"
            else:
                props = torch.cuda.get_device_properties(device)
                name = f"{props.name} ({total_gb:.1f}GB total, {free_gb:.1f}GB free)"
    elif torch.backends.mps.is_available():
        device, name = torch.device("mps"), "Apple MPS"
    else:
        device, name = torch.device("cpu"), "CPU"

    if verbose:
        print(f"Device: [{device}] {name}")
    return device
