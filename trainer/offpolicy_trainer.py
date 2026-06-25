"""Off-policy and on-policy TEMP trainers.

Two classes:

* :class:`OffPolicyTrainer` — SAC-style trainer where the **env and replay buffer
  live in the trainer**, not the policy. Steps through the env one transition at a
  time, pushes to the buffer, then calls ``policy.update_sac(batch)`` for every
  new sample (UTD ratio applied). Used for TEMP with ``optimal_policy="sac"``.

* :class:`OnPolicyTEMPTrainer` — on-policy trainer for TEMP with
  ``optimal_policy="ppo"``. Runs the :class:`OnlineSampler` **twice** per
  iteration — once for the contracting actor (γ→0) and once for the optimal
  actor (high γ) — then calls ``policy.update_ppo(con_batch, opt_batch)``.
"""

import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base
from trainer.evaluator import Evaluator
from utils.replay_buffer import ReplayBuffer
from utils.sampler import OnlineSampler


class OffPolicyTrainer(Evaluator):
    """SAC off-policy trainer: env + replay buffer owned by the trainer.

    At each env step:
      1. Deploy ``policy.actor`` (the optimal actor) to collect one transition.
      2. Push to replay buffer.
      3. Once ``buffer.size >= learning_starts``: sample ``utd_ratio`` minibatches
         and call ``policy.update_sac(s, a, ns, term)`` for each.
      4. Every ``cmg_update_freq`` SAC updates, call ``policy.update_cmg()``
         ``cmg_updates_per_iter`` times.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        epochs: int = int(1e6),
        log_interval: int = 1000,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
        rendering: bool = False,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 5000,
        utd_ratio: float = 1.0,
        cmg_update_freq: int = 1,
        cmg_updates_per_iter: int = 1,
    ):
        super().__init__(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=epochs,
            log_interval=log_interval,
            eval_num=eval_num,
            eval_episodes=eval_episodes,
            seed=seed,
            rendering=rendering,
        )
        self.learning_starts = learning_starts
        self.utd_ratio = utd_ratio
        self.cmg_update_freq = cmg_update_freq
        self.cmg_updates_per_iter = cmg_updates_per_iter

        self.buffer = ReplayBuffer(
            state_dim=policy.state_dim,
            u_dim=policy.u_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=torch.device(policy.device),
        )

    def _step_env(self, obs: np.ndarray):
        """One env step using the deployed (optimal) actor."""
        with torch.no_grad():
            a_t, _ = self.policy(obs)   # forward() accepts numpy
        a_np = a_t.cpu().numpy().squeeze(0)
        if a_np.ndim == 0:
            a_np = a_np.reshape(1)
        next_obs, env_rew, term, trunc, _ = self.env.step(a_np)
        done = term or trunc
        self.buffer.direct_append(
            states=obs[None],
            controls=a_np[None],
            next_states=next_obs[None],
            rewards=np.array([[float(env_rew)]], dtype=np.float32),
            terminations=np.array([[float(term)]], dtype=np.float32),
        )
        return next_obs, done, float(env_rew)

    def train(self):
        start_time = time.time()
        self.last_perf_score = deque(maxlen=1)

        obs, _ = self.env.reset(seed=self.seed)
        total_timesteps = self.epochs + self.init_epochs
        eval_idx = 0
        sac_updates_total = 0
        loss_agg = {}

        # --- warmup: fill the replay buffer before any policy update ---
        # No step counting or per-step logging happens before the first update;
        # the whole warmup collection is surfaced as a single wandb logging tick.
        self.policy.train()
        while self.buffer.size < self.learning_starts:
            next_obs, done, _ = self._step_env(obs)
            obs = self.env.reset(seed=self.seed)[0] if done else next_obs
        if self.learning_starts > 0:
            self.write_log(
                {f"{self.policy.name}/RL_analytics/buffer_size": self.buffer.size},
                step=self.init_epochs,
            )

        with tqdm(
            initial=self.init_epochs,
            total=total_timesteps,
            desc=f"{self.policy.name} Training (Epochs)",
        ) as pbar:
            while pbar.n < total_timesteps:
                step = pbar.n + 1
                self.policy.progress = step / total_timesteps

                self.policy.train()

                # --- collect one env transition ---
                next_obs, done, env_rew = self._step_env(obs)
                obs = self.env.reset(seed=self.seed)[0] if done else next_obs

                loss_dict = {
                    f"{self.policy.name}/RL_analytics/buffer_size": self.buffer.size,
                    f"{self.policy.name}/RL_analytics/avg_env_reward": env_rew,
                }

                # --- off-policy updates (buffer is already warm) ---
                n_updates = max(1, int(self.utd_ratio))
                for _ in range(n_updates):
                    s, c, ns, _, term_t = self.buffer.sample()
                    info = self.policy.update_sac(s, c, ns, term_t)
                    for k, v in info.items():
                        loss_agg.setdefault(k, []).append(v)
                    sac_updates_total += 1

                    if sac_updates_total % self.cmg_update_freq == 0:
                        for _ in range(self.cmg_updates_per_iter):
                            cmg_info = self.policy.update_cmg()
                            for k, v in cmg_info.items():
                                loss_agg.setdefault(k, []).append(v)

                loss_dict[f"{self.policy.name}/RL_analytics/n_updates"] = n_updates

                # flush aggregated losses every log_interval steps
                if step % self.log_interval == 0 and loss_agg:
                    for k, vs in loss_agg.items():
                        loss_dict[k] = float(np.mean(vs))
                    loss_agg = {}

                pbar.update(n_updates)

                self.write_log(loss_dict, step=step)

                # --- periodic evaluation ---
                if step >= self.eval_interval * eval_idx:
                    self.policy.eval()
                    eval_dict, supp_dict = self.evaluate()
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(supp_dict, step=step)
                    self.last_perf_score.append(eval_dict["eval/performance_score"])
                    self.save_model(step)
                    self.policy.train()
                    eval_idx += 1

            # final evaluation
            self.policy.eval()
            eval_dict, supp_dict = self.evaluate()
            self.write_log(eval_dict, step=pbar.n, eval_log=True)
            self.save_model(pbar.n)
            torch.cuda.empty_cache()

        self.logger.print(
            "total training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def save_model(self, e):
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)
        model = (
            getattr(self.policy, "u_func", None)
            or getattr(self.policy, "actor", None)
            or self.policy
        )
        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)
            if np.mean(self.last_perf_score) > self.last_max_perf_score:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.logger.log_dir, "best_model.pth"),
                )
                self.last_max_perf_score = np.mean(self.last_perf_score)
        else:
            raise ValueError("Model is not identifiable!")


class OnPolicyTEMPTrainer(Evaluator):
    """On-policy TEMP trainer for PPO mode.

    Runs the :class:`OnlineSampler` twice per iteration — once with
    ``policy.con_actor`` (γ→0 contracting) and once with ``policy.opt_actor``
    (high-γ optimal) — then calls ``policy.update_ppo(con_batch, opt_batch)``
    followed by ``policy.update_cmg()`` × ``cmg_updates_per_iter``.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        timesteps: int = int(1e6),
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
        rendering: bool = False,
        cmg_updates_per_iter: int = 1,
    ):
        super().__init__(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=timesteps,
            log_interval=log_interval,
            eval_num=eval_num,
            eval_episodes=eval_episodes,
            seed=seed,
            rendering=rendering,
        )
        self.sampler = sampler
        self.cmg_updates_per_iter = cmg_updates_per_iter

    def train(self):
        start_time = time.time()
        self.last_perf_score = deque(maxlen=1)

        total_timesteps = self.epochs + self.init_epochs
        eval_idx = 0

        with tqdm(
            initial=self.init_epochs,
            total=total_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_timesteps:
                step = pbar.n + 1
                progress = step / total_timesteps
                self.policy.progress = progress

                self.policy.train()

                # --- collect from contracting policy (γ→0) ---
                self.policy.actor = self.policy.con_actor
                con_batch, con_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )

                # --- collect from optimal policy (only when not con_only) ---
                con_only = getattr(self.policy, "con_only", False)
                if not con_only:
                    self.policy.actor = self.policy.opt_actor
                    opt_batch, opt_time = self.sampler.collect_samples(
                        env=self.env, policy=self.policy, seed=self.seed
                    )
                else:
                    opt_batch, opt_time = None, 0.0

                # --- PPO update ---
                loss_dict = self.policy.update_ppo(con_batch, opt_batch, progress)

                # --- CMG update ---
                for _ in range(self.cmg_updates_per_iter):
                    cmg_info = self.policy.update_cmg()
                    for k, v in cmg_info.items():
                        loss_dict[k] = v

                n_steps = con_batch["rewards"].shape[0]
                loss_dict[f"{self.policy.name}/RL_analytics/n_con_steps"] = n_steps
                loss_dict[f"{self.policy.name}/RL_analytics/sample_time"] = (
                    con_time + opt_time
                )

                pbar.update(n_steps)

                self.write_log(loss_dict, step=step)

                # --- periodic evaluation ---
                if step >= self.eval_interval * eval_idx:
                    self.policy.eval()
                    eval_dict, supp_dict = self.evaluate()
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(supp_dict, step=step)
                    self.last_perf_score.append(eval_dict["eval/performance_score"])
                    self.save_model(step)
                    self.policy.train()
                    eval_idx += 1

            # final evaluation
            self.policy.eval()
            eval_dict, supp_dict = self.evaluate()
            self.write_log(eval_dict, step=pbar.n, eval_log=True)
            self.save_model(pbar.n)
            torch.cuda.empty_cache()

        self.logger.print(
            "total training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def save_model(self, e):
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)
        model = (
            getattr(self.policy, "u_func", None)
            or getattr(self.policy, "actor", None)
            or self.policy
        )
        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)
            if np.mean(self.last_perf_score) > self.last_max_perf_score:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.logger.log_dir, "best_model.pth"),
                )
                self.last_max_perf_score = np.mean(self.last_perf_score)
        else:
            raise ValueError("Model is not identifiable!")
