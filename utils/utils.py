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
from policy.cpo import CPO
from policy.layers.CMG_networks import CCM_Generator
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

    args.state_dim = env.observation_space.shape[0]
    args.x_dim = env.num_dim_x
    args.u_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len

    return env


from policy import C3M, CARL, LQR, PPO, SD_LQR


def _create_actor_critic(args):
    """Helper to instantiate Actor and Critic based on policy type."""
    if args.policy_type == "CL":
        actor = CLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            num_windows=args.num_windows,
            mode=args.policy_mode,
            anneal_stddev=args.anneal_stddev,
        )
        critic = RLCritic(args.state_dim, hidden_dim=args.critic_dim)
    elif args.policy_type == "RL":
        actor = RLActor(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            hidden_dim=args.actor_dim,
            mode=args.policy_mode,
            anneal_stddev=args.anneal_stddev,
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


def _create_cmg(x_dim: int, mode: str, device: torch.device) -> CCM_Generator:
    """Helper to create the CCM Generator."""
    return CCM_Generator(
        x_dim=x_dim,
        hidden_dim=[128, 128],
        activation=nn.Tanh(),
        mode=mode,
        device=device,
    )


def get_policy(env, args, get_f_and_B, SDC_func=None):
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
            K=args.K_epochs,
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
        CMG = _create_cmg(args.x_dim, mode="deterministic", device=args.device)
        # C3M uses a specific deterministic actor
        actor = CLActor(x_dim=args.x_dim, u_dim=args.u_dim, mode="deterministic")
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
            device=args.device,
        )

    elif algo == "carl":
        CMG = _create_cmg(args.x_dim, mode=args.CMG_mode, device=args.device)
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
            disable_CMG_training=args.disable_CMG_training,  # Note the inversion here
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
            K=args.K_epochs,
            nupdates=nupdates,
            policy_updates_per_cmg_update=args.policy_updates_per_cmg_update,
            device=args.device,
        )

    elif algo == "cpo":
        CMG = _create_cmg(args.x_dim, mode=args.CMG_mode, device=args.device)
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
