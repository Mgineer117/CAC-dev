import numpy as np
import torch
import torch.nn as nn

from envs import (
    CarEnv,
    CartPoleEnv,
    FlapperEnv,
    NeuralLanderEnv,
    PvtolEnv,
    QuadRotorEnv,
    SegwayEnv,
    TurtlebotEnv,
)
from policy import C3M, CARL, CARL_M, CORL, LQR, NCM, PPO, SD_LQR
from policy.cpo import CPO
from policy.layers.CMG_networks_bounded import BoundedCCM_Generator
from policy.layers.policy_networks import (
    CLActor,
    EncoderCLActor,
    EncoderRLCritic,
    RLActor,
    RLCritic,
)
from policy.trpo import TRPO


def get_env(args):
    # 1. Define the map of strings to Class objects
    env_map = {
        "car": CarEnv,
        "pvtol": PvtolEnv,
        "quadrotor": QuadRotorEnv,
        "neurallander": NeuralLanderEnv,
        "segway": SegwayEnv,
        "turtlebot": TurtlebotEnv,
        "cartpole": CartPoleEnv,
        "flapper": FlapperEnv,
    }

    # 2. Check existence
    if args.task not in env_map:
        raise NotImplementedError(f"{args.task} is not implemented.")

    # 3. Instantiate once using the common arguments
    env = env_map[args.task](
        sample_mode=args.sample_mode,
        reward_mode=args.reward_mode,
        num_windows=args.num_windows,
    )
    env.lbd = args.lbd
    env.gamma = args.gamma

    if getattr(args, "control_scaler", None) is not None:
        env.control_scaler = args.control_scaler

    args.state_dim = env.observation_space.shape[0]
    args.x_dim = env.num_dim_x
    args.u_dim = env.action_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len

    return env


def _create_actor_critic(args):
    """Helper to instantiate Actor and Critic based on policy type."""
    actor_activation = getattr(args, "actor_activation", "tanh")
    if args.policy_type == "CL":
        actor = CLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            num_windows=args.num_windows,
            mode=args.policy_mode,
            anneal_stddev=args.anneal_stddev,
            hidden_dim=args.actor_dim,
            activation=actor_activation,
        )
        critic = RLCritic(args.state_dim, hidden_dim=args.critic_dim)
    elif args.policy_type == "RL":
        actor = RLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            hidden_dim=args.actor_dim,
            mode=args.policy_mode,
            anneal_stddev=args.anneal_stddev,
            activation=actor_activation,
        )
        critic = RLCritic(args.state_dim, hidden_dim=args.critic_dim)
    elif args.policy_type == "EncoderCL":
        actor = EncoderCLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            latent_dim=args.x_dim,
            num_windows=args.num_windows,
            mode=args.policy_mode,
            anneal_stddev=args.anneal_stddev,
        )
        critic = EncoderRLCritic(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            latent_dim=args.x_dim,
            num_windows=args.num_windows,
            hidden_dim=args.critic_dim,
        )
    else:
        raise ValueError(f"Unknown policy_type: {args.policy_type}")
    return actor, critic


def _create_cmg(args, mode: str, device: torch.device):
    """Create the contraction metric generator.

    Always the BoundedCCM_Generator: an eigenvalue sigmoid enforces the
    w_lb/w_ub bounds by construction (the strict matrix analogue of a*tanh for
    actions), so no overshoot loss or w_lb*I shift is needed downstream.
    """
    cmg_hidden_dims = getattr(args, "cmg_hidden_dims", [128, 128])
    cmg_activation = getattr(args, "cmg_activation", "tanh")
    return BoundedCCM_Generator(
        x_dim=args.x_dim,
        hidden_dim=cmg_hidden_dims,
        activation=cmg_activation,
        mode=mode,
        w_lb=args.w_lb if args.w_lb is not None else 0.1,
        w_ub=args.w_ub if args.w_ub is not None else 10.0,
        device=device,
    )


def get_policy(env, args, get_f_and_B, SDC_func=None, logger=None, writer=None):
    algo = args.algo_name
    nupdates = args.timesteps / (args.minibatch_size * args.num_minibatch)
    if args.gamma is not None:
        gamma = args.gamma
    else:
        gamma = env.get_horizon_matched_gamma()

    # --- 1. LQR Family ---
    if algo.startswith(("lqr", "sd-lqr")):
        if algo.startswith("lqr"):
            return LQR(x_dim=args.x_dim, action_dim=args.u_dim, get_f_and_B=get_f_and_B)
        else:
            return SD_LQR(
                x_dim=args.x_dim,
                action_dim=args.u_dim,
                get_f_and_B=get_f_and_B,
                SDC_func=SDC_func,
            )

    # --- 2. PPO ---
    elif algo.startswith("ppo"):
        actor, critic = _create_actor_critic(args)
        return PPO(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            latent_dim=args.x_dim,
            num_windows=args.num_windows,
            actor=actor,
            critic=critic,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=gamma,
            gae=args.gae,
            K=args.k_epochs,
            nupdates=nupdates,
            device=args.device,
        )
    elif algo.startswith("trpo"):
        actor, critic = _create_actor_critic(args)
        return TRPO(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            latent_dim=args.x_dim,
            num_windows=args.num_windows,
            actor=actor,
            critic=critic,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            target_kl=args.target_kl,
            gamma=gamma,
            gae=args.gae,
            nupdates=nupdates,
            device=args.device,
        )

    # --- 3. C3M Family ---
    elif algo == "c3m":
        CMG = _create_cmg(args, mode="deterministic", device=args.device)
        # C3M uses a specific deterministic actor
        actor = CLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            mode="deterministic",
            hidden_dim=args.actor_dim,
            activation=getattr(args, "actor_activation", "tanh"),
        )
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        return C3M(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            CMG=CMG,
            actor=actor,
            data=data,
            get_f_and_B=get_f_and_B,
            W_lr=args.W_lr,
            u_lr=args.u_lr,
            lbd=args.lbd,
            eps=args.eps,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            nupdates=args.c3m_epochs,
            # optional SD-LQR CMG pretraining (the CORL recipe); reuses the corl-* args
            pretrain_cmg=args.c3m_pretrain_cmg,
            pretrain_c1c2=getattr(args, "c3m_pretrain_c1c2", False),
            SDC_func=SDC_func,
            Q_scaler=args.Q_scaler,
            R_scaler=args.R_scaler,
            pretrain_epochs=args.corl_pretrain_epochs,
            pretrain_buffer_size=args.corl_pretrain_buffer_size,
            pretrain_minibatch_size=args.corl_pretrain_minibatch_size,
            pretrain_W_lr=args.corl_pretrain_W_lr,
            val_split=args.corl_val_split,
            val_interval=args.corl_val_interval,
            plateau_tol=args.corl_plateau_tol,
            plateau_patience=args.corl_plateau_patience,
            logger=logger,
            writer=writer,
            device=args.device,
        )

    elif algo == "carl":
        CMG = _create_cmg(args, mode=args.cmg_mode, device=args.device)
        actor, critic = _create_actor_critic(args)
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        return CARL(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            CMG=CMG,
            get_f_and_B=get_f_and_B,
            data=data,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            disable_CMG_training=args.disable_cmg_training,  # Note the inversion here
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            W_entropy_scaler=args.W_entropy_scaler,
            reward_mode=args.reward_mode,
            entropy_scaler=args.entropy_scaler,
            tracking_scaler=env.tracking_scaler,
            control_scaler=env.control_scaler,
            target_kl=args.target_kl,
            num_windows=args.num_windows,
            gamma=gamma,
            gae=args.gae,
            K=args.k_epochs,
            nupdates=nupdates,
            policy_updates_per_cmg_update=args.policy_updates_per_cmg_update,
            device=args.device,
        )

    elif algo == "carl_m":
        # CARL with the raw Mahalanobis tracking reward -||e||^2_M.
        # reward_mode is hardcoded to "mahal" inside CARL_M; no need to pass it.
        CMG = _create_cmg(args, mode=args.cmg_mode, device=args.device)
        actor, critic = _create_actor_critic(args)
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        return CARL_M(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            CMG=CMG,
            get_f_and_B=get_f_and_B,
            data=data,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            disable_CMG_training=args.disable_cmg_training,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            W_entropy_scaler=args.W_entropy_scaler,
            entropy_scaler=args.entropy_scaler,
            tracking_scaler=env.tracking_scaler,
            control_scaler=env.control_scaler,
            target_kl=args.target_kl,
            num_windows=args.num_windows,
            gamma=gamma,
            gae=args.gae,
            K=args.k_epochs,
            nupdates=nupdates,
            policy_updates_per_cmg_update=args.policy_updates_per_cmg_update,
            device=args.device,
        )

    elif algo == "ncm":
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        alpha = args.cvstem_alpha if args.cvstem_alpha is not None else args.lbd
        return NCM(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            data=data,
            get_f_and_B=get_f_and_B,
            dt=args.cvstem_dt if args.cvstem_dt is not None else env.dt,
            alpha=alpha,
            w_nu=args.cvstem_w_nu,
            R_scaler=args.ncm_R_scaler,
            epsilon=args.eps,
            linesearch=not args.cvstem_no_linesearch,
            include_dwdt=not args.cvstem_no_dwdt,
            hidden_dims=args.cmg_hidden_dims,
            activation=args.cmg_activation,
            w_lb=args.w_lb,
            W_lr=args.W_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            cvstem_num_samples=args.cvstem_num_samples,
            nupdates=args.c3m_epochs,
            num_windows=args.num_windows,
            device=args.device,
            logger=logger,
            writer=writer,
        )

    elif algo == "corl":
        CMG = _create_cmg(args, mode=args.cmg_mode, device=args.device)
        actor, critic = _create_actor_critic(args)
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        return CORL(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            CMG=CMG,
            get_f_and_B=get_f_and_B,
            data=data,
            actor=actor,
            critic=critic,
            SDC_func=SDC_func,
            Q_scaler=args.Q_scaler,
            R_scaler=args.R_scaler,
            pretrain_epochs=args.corl_pretrain_epochs,
            pretrain_buffer_size=args.corl_pretrain_buffer_size,
            pretrain_minibatch_size=args.corl_pretrain_minibatch_size,
            pretrain_W_lr=args.corl_pretrain_W_lr,
            val_split=args.corl_val_split,
            val_interval=args.corl_val_interval,
            plateau_tol=args.corl_plateau_tol,
            plateau_patience=args.corl_plateau_patience,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            W_entropy_scaler=args.W_entropy_scaler,
            entropy_scaler=args.entropy_scaler,
            tracking_scaler=env.tracking_scaler,
            control_scaler=env.control_scaler,
            target_kl=args.target_kl,
            num_windows=args.num_windows,
            gamma=gamma,
            gae=args.gae,
            K=args.k_epochs,
            nupdates=nupdates,
            policy_updates_per_cmg_update=args.policy_updates_per_cmg_update,
            logger=logger,
            writer=writer,
            device=args.device,
        )

    elif algo == "cpo":
        CMG = _create_cmg(args, mode=args.cmg_mode, device=args.device)
        actor, critic = _create_actor_critic(args)
        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")

        return CPO(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            dt=env.dt,
            data=data,
            actor=actor,
            critic=critic,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            lbd=args.lbd,
            reward_mode=args.reward_mode,
            tracking_scaler=env.tracking_scaler,
            control_scaler=env.control_scaler,
            target_kl=args.target_kl,
            num_windows=args.num_windows,
            gamma=gamma,
            gae=args.gae,
            nupdates=nupdates,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
