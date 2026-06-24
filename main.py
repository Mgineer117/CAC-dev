# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #

import datetime
import random
import uuid

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from trainer.c3m_trainer import C3MTrainer
from trainer.offpolicy_trainer import OffPolicyTrainer, OnPolicyTEMPTrainer
from trainer.online_trainer import OnlineTrainer
from utils.get_args import get_args
from utils.get_dynamics import get_dynamics
from utils.get_sdc import get_SDC
from utils.misc import (
    concat_csv_columnwise_and_delete,
    override_args,
    seed_all,
    setup_logger,
)
from utils.sampler import OnlineSampler
from utils.utils import get_env, get_policy


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    env = get_env(args)  # can use approximated dynamics
    eval_env = get_env(args)  # always uses true dynamics
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    # get dynamics and use it for simulation
    get_f_and_B, init_epochs = get_dynamics(env, args, logger, writer)
    # get SDC
    SDC_func, init_epochs = get_SDC(env, args, logger, writer, get_f_and_B, init_epochs)

    policy = get_policy(eval_env, args, get_f_and_B, SDC_func, logger=logger, writer=writer)

    if hasattr(policy, "warmup_result"):
        logger.write_images(
            step=0, image=policy.warmup_result, logdir="CMG_warmup_result"
        )

    # Keep the shared wandb global step monotonic: some policies (e.g. CORL) stream
    # pretraining curves to wandb during get_policy, advancing the global step. The
    # online/offline trainer logs with explicit steps based on init_epochs, so we
    # must start it above the current wandb step or its early logs get dropped.
    if wandb.run is not None:
        try:
            init_epochs = max(init_epochs, int(wandb.run.step))
        except Exception:
            pass

    if args.algo_name in ("temp", "temp2"):
        if args.temp_optimal_policy == "sac":
            trainer = OffPolicyTrainer(
                env=env,
                eval_env=eval_env,
                policy=policy,
                logger=logger,
                writer=writer,
                init_epochs=init_epochs,
                epochs=args.epochs,
                log_interval=args.log_interval,
                eval_num=args.eval_num,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                rendering=args.rendering,
                buffer_size=args.sac_buffer_size,
                batch_size=args.sac_batch_size,
                learning_starts=args.sac_learning_starts,
                utd_ratio=args.sac_utd,
                cmg_update_freq=int(args.minibatch_size * args.num_minibatch),
                cmg_updates_per_iter=args.temp_cmg_updates_per_iter,
            )
        else:  # ppo
            sampler = OnlineSampler(
                state_dim=args.state_dim,
                u_dim=args.u_dim,
                episode_len=args.episode_len,
                batch_size=int(args.minibatch_size * args.num_minibatch),
            )
            trainer = OnPolicyTEMPTrainer(
                env=env,
                eval_env=eval_env,
                policy=policy,
                sampler=sampler,
                logger=logger,
                writer=writer,
                init_epochs=init_epochs,
                timesteps=args.timesteps,
                log_interval=args.log_interval,
                eval_num=args.eval_num,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                rendering=args.rendering,
                cmg_updates_per_iter=args.temp_cmg_updates_per_iter,
            )
        trainer.train()
    elif args.algo_name.startswith(("carl", "sac", "ppo", "trpo", "cpo")):
        sampler = OnlineSampler(
            state_dim=args.state_dim,
            u_dim=args.u_dim,
            episode_len=args.episode_len,
            batch_size=int(args.minibatch_size * args.num_minibatch),
        )

        trainer = OnlineTrainer(
            env=env,
            eval_env=eval_env,
            policy=policy,
            sampler=sampler,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            timesteps=args.timesteps,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            rendering=args.rendering,
        )
        trainer.train()
    elif args.algo_name.startswith(("c3m", "ncm")):
        trainer = C3MTrainer(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=args.epochs,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            rendering=args.rendering,
        )
        trainer.train()
    else:
        from trainer.evaluator import Evaluator

        evaluator = Evaluator(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=args.timesteps,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            rendering=args.rendering,
        )
        evaluator.begin_evaluate()

    wandb.finish()


if __name__ == "__main__":
    # initialization
    torch.set_default_dtype(torch.float32)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed

        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
