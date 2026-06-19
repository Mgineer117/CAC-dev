import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--task",
        type=str,
        default="car",
        help="Define the task = [car, pvtol, neurallander, quadrotor].",
    )
    parser.add_argument("--algo-name", type=str, default="cac", help="Algorithm name.")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default=None,
        help='Policy mode: ["deterministic", "stochastic"].',
    )
    parser.add_argument(
        "--disable-CMG-training",  # Often named with "no-" to make it intuitive
        action="store_true",
        help="Disable CMG training (default: True)",
    )
    parser.add_argument(
        "--CMG-mode",
        type=str,
        default=None,
        help='CMG mode: ["deterministic", "stochastic"].',
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help='Policy type: ["EncoderCL", "CL", "RL"].',
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of experiments for each algorithm.",
    )
    parser.add_argument(
        "--actor-lr", type=float, default=3e-4, help="Actor learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4, help="Critic learning rate."
    )
    parser.add_argument(
        "--Dynamic-lr",
        type=float,
        default=1e-3,
        help="Dynamic approximator learning rate.",
    )
    parser.add_argument(
        "--SDC-lr",
        type=float,
        default=1e-3,
        help="SDC decomposition neural net learning rate.",
    )

    parser.add_argument("--W-lr", type=float, default=None, help="CMG learning rate.")
    parser.add_argument(
        "--u-lr", type=float, default=None, help="C3M actor learning rate."
    )
    parser.add_argument(
        "--w-ub", type=float, default=None, help="Contraction metric upper bound."
    )
    parser.add_argument(
        "--w-lb", type=float, default=None, help="Contraction metric lower bound."
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.1, help="Epsilon clip for PPO."
    )
    parser.add_argument(
        "--eps", type=float, default=0.1, help="Used for CMG learning regularization."
    )
    parser.add_argument(
        "--lbd", type=float, default=None, help="Desired contraction rate."
    )
    parser.add_argument(
        "--policy-updates-per-cmg-update",
        type=int,
        default=5,
        help="Number of policy updates per CMG update.",
    )
    # --- NCM / CV-STEM (Tsukamoto, exact convex SDP) ---
    parser.add_argument(
        "--cvstem-alpha",
        type=float,
        default=None,
        help="NCM: CV-STEM contraction rate alpha (defaults to --lbd; only used "
        "when --cvstem-no-linesearch is set).",
    )
    parser.add_argument(
        "--ncm-R-scaler",
        type=float,
        default=1.0,
        help="NCM: control weight R = R_scaler*I for the u = u* - R^-1 B^T M e law.",
    )
    parser.add_argument(
        "--cvstem-dt",
        type=float,
        default=None,
        help="NCM: dt used in the CV-STEM (W-I)/dt term (defaults to env.dt). A "
        "larger value relaxes the time-derivative bound.",
    )
    parser.add_argument(
        "--cvstem-w-nu",
        type=float,
        default=1.0,
        help="NCM: weight on nu (control authority) in the CV-STEM objective.",
    )
    parser.add_argument(
        "--cvstem-num-samples",
        type=int,
        default=100,
        help="NCM: number of states in the joint CV-STEM SDP (kept modest; the "
        "NCM network interpolates between them).",
    )
    parser.add_argument(
        "--cvstem-no-linesearch",
        action="store_true",
        help="NCM: disable the alpha line search and use --cvstem-alpha / --lbd.",
    )
    parser.add_argument(
        "--cvstem-no-dwdt",
        action="store_true",
        help="NCM: drop the (W-I)/dt term (steady-state / constant-metric CV-STEM; "
        "better conditioned when dt is tiny).",
    )
    # --- CORL (SD-LQR pretrained CMG) ---
    parser.add_argument(
        "--Q-scaler",
        type=float,
        default=1.0,
        help="CORL: state cost scaler for the SD-LQR Riccati equation.",
    )
    parser.add_argument(
        "--R-scaler",
        type=float,
        default=0.0,
        help="CORL: control cost scaler for the SD-LQR Riccati equation.",
    )
    parser.add_argument(
        "--corl-pretrain-epochs",
        type=int,
        default=5000,
        help="CORL: max number of CMG pretraining epochs (early stopping may end it sooner).",
    )
    parser.add_argument(
        "--corl-pretrain-buffer-size",
        type=int,
        default=10000,
        help="CORL: number of states for which SD-LQR controls are precomputed.",
    )
    parser.add_argument(
        "--corl-pretrain-minibatch-size",
        type=int,
        default=1024,
        help="CORL: minibatch size used during CMG pretraining.",
    )
    parser.add_argument(
        "--corl-pretrain-W-lr",
        type=float,
        default=1e-3,
        help="CORL: initial CMG learning rate for pretraining (cosine-annealed to 0).",
    )
    parser.add_argument(
        "--corl-val-split",
        type=float,
        default=0.1,
        help="CORL: held-out fraction of pretrain states for early-stopping validation.",
    )
    parser.add_argument(
        "--corl-val-interval",
        type=int,
        default=25,
        help="CORL: epochs between validation-loss evaluations during pretraining.",
    )
    parser.add_argument(
        "--corl-plateau-tol",
        type=float,
        default=1e-3,
        help="CORL: relative moving-average change below which the val loss is a plateau.",
    )
    parser.add_argument(
        "--corl-plateau-patience",
        type=int,
        default=3,
        help="CORL: consecutive plateau checks required to early-stop pretraining.",
    )
    parser.add_argument(
        "--DynamicLearner-dim",
        type=list,
        default=[256, 256],
        help="Dynamic approximator hidden layer.",
    )
    parser.add_argument(
        "--cmg-hidden-dims",
        type=list,
        default=[128, 128],
        help="CMG network hidden layer dimensions.",
    )
    parser.add_argument(
        "--cmg-activation",
        type=str,
        default="tanh",
        help="CMG network activation function ['tanh', 'relu', 'siren'].",
    )
    parser.add_argument(
        "--SDCLearner-dim",
        type=list,
        default=[256, 256],
        help="SDC decomposition neural net hidden layer.",
    )
    parser.add_argument(
        "--actor-dim", type=list, default=[64, 64], help="actor hidden layers."
    )
    parser.add_argument(
        "--actor-activation",
        type=str,
        default="tanh",
        help="actor activation ['tanh', 'relu', 'elu', 'leaky_relu', 'gelu'].",
    )
    parser.add_argument(
        "--critic-dim", type=list, default=[256, 256], help="critic hidden layers."
    )

    parser.add_argument(
        "--c3m-epochs", type=int, default=None, help="Number of training samples."
    )
    parser.add_argument(
        "--dynamics-epochs",
        type=int,
        default=20000,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--sdc-epochs",
        type=int,
        default=2000,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Number of training samples."
    )
    parser.add_argument(
        "--num-windows", type=int, default=None, help="Number of training samples."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Number of evaluation throughout timesteps.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per reference trajectory.",
    )
    parser.add_argument(
        "--eval-num",
        type=int,
        default=10,
        help="Number of reference trajectory for evaluation.",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="Disturbance rate.")
    parser.add_argument(
        "--c3m-buffer-size", type=int, default=200_000, help="Number of mini-batches."
    )
    parser.add_argument(
        "--dynamics-buffer-size",
        type=int,
        default=100_000,
        help="Number of mini-batches.",
    )
    parser.add_argument(
        "--sdc-buffer-size",
        type=int,
        default=100_000,
        help="Number of mini-batches.",
    )
    parser.add_argument(
        "--num-minibatch", type=int, default=4, help="Number of mini-batches."
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=1024, help="Size of each mini-batch."
    )
    parser.add_argument(
        "--K-epochs", type=int, default=5, help="Number of K epochs in PPO."
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.003,
        help="Target KL divergence.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation factor.",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="Uniform",
        help="Sampling mode for generating offline data for learning dynamics.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        help="Reward mode for the environment.",
    )
    parser.add_argument(
        "--anneal-stddev", action="store_true", default=None, help="Anneal stddev during training."
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=0.0, help="Entropy scaling factor."
    )
    parser.add_argument(
        "--W-entropy-scaler", type=float, default=None, help="W entropy scaling factor."
    )
    parser.add_argument(
        "--control-scaler",
        type=float,
        default=None,
        help="Control scaling factor to reward.",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument(
        "--load-pretrained-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    parser.add_argument("--gpu-idx", type=int, default=0, help="GPU index.")
    parser.add_argument("--rendering", action="store_true", help="Render environment as video to wandb.")

    args = parser.parse_args()

    import json
    import os

    # Load task config
    task_config_path = os.path.join("config", "task", f"{args.task}.json")
    if os.path.exists(task_config_path):
        with open(task_config_path, "r") as f:
            task_config = json.load(f)
        for k, v in task_config.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    # Load algo config
    algo_config_path = os.path.join("config", "algorithm", f"{args.algo_name}.json")
    if os.path.exists(algo_config_path):
        with open(algo_config_path, "r") as f:
            algo_config = json.load(f)
        for k, v in algo_config.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    args.device = select_device(args.gpu_idx)

    return args


import torch


def select_device(device_arg=0, verbose=True):
    """
    Selects the best available device (CUDA > MPS > CPU) with robust error handling.

    Args:
        device_arg (int | str | None): GPU index (e.g., 0), 'cpu', or None for auto.
    """
    # 1. Force CPU if requested
    if device_arg == "cpu":
        device = torch.device("cpu")
        device_name = "CPU (Forced)"

    # 2. Check CUDA (NVIDIA)
    elif torch.cuda.is_available():
        # Handle auto-selection (None) defaulting to 0
        idx = 0 if device_arg is None else int(device_arg)

        # Safe check: Does this GPU index exist?
        if idx < torch.cuda.device_count():
            device = torch.device(f"cuda:{idx}")
            torch.cuda.empty_cache()  # Clear cache for a fresh start

            # Get cool stats
            props = torch.cuda.get_device_properties(device)
            vram = props.total_memory / 1e9  # Convert to GB
            device_name = f"{props.name} ({vram:.1f}GB VRAM)"
        else:
            # Fallback if index is out of bounds
            print(
                f"⚠️  Warning: GPU {idx} requested but only {torch.cuda.device_count()} found. Switching to CPU."
            )
            device = torch.device("cpu")
            device_name = "CPU (Fallback)"

    # 3. Check MPS (Apple Silicon - M1/M2/M3)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS (Metal Performance Shaders)"

    # 4. Default to CPU
    else:
        device = torch.device("cpu")
        device_name = "CPU (Standard)"

    # --- Cool Output ---
    if verbose:
        print("=" * 60)
        status = "🟢 ON" if device.type != "cpu" else "🟡 ON"
        print(f"🚀 Device Selected: {status}  [{device}]")
        print(f"ℹ️  Details        : {device_name}")
        print("=" * 60)

    return device
