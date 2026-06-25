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
from utils.sampler import OnlineSampler


# model-free policy trainer
class OnlineTrainer(Evaluator):
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
        rendering: bool = False,
    ) -> None:
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

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_perf_score = deque(maxlen=1)

        # Define a helper function to avoid code duplication
        def run_eval(step_count, eval_log=True):
            self.policy.eval()
            eval_dict, supp_dict = self.evaluate()

            # Manual logging
            self.write_log(eval_dict, step=step_count, eval_log=eval_log)
            self.write_image(supp_dict, step=step_count)
            self.last_perf_score.append(eval_dict["eval/performance_score"])
            self.save_model(step_count)

            # Return policy to train mode after eval
            self.policy.train()

        # Train loop
        eval_idx = 0
        total_timesteps = self.epochs + self.init_epochs

        with tqdm(
            initial=self.init_epochs,
            total=total_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division

                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )

                loss_dict, supp_dict, update_time = self.policy.learn(
                    batch, progress=step / total_timesteps
                )

                n_steps = batch["rewards"].shape[0]
                n_updates = int(loss_dict.get(
                    f"{self.policy.name}/RL_analytics/n_updates", n_steps
                ))

                # Warmup (off-policy buffer not yet filled): no policy update
                # happened, so don't count it as a training step or run eval —
                # just surface a single wandb logging tick and keep collecting.
                if n_updates == 0:
                    self.write_log(loss_dict, step=step)
                    continue

                pbar.update(n_updates)

                loss_dict[f"{self.policy.name}/RL_analytics/timesteps"] = step
                loss_dict[f"{self.policy.name}/RL_analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/RL_analytics/update_time"] = update_time

                self.write_log(loss_dict, step=step)
                self.write_image(
                    supp_dict,
                    step=step,
                )

                #### PERIODIC EVALUATIONS ####
                # Only check for the interval here. We removed the "OR" condition.
                if step >= self.eval_interval * eval_idx:
                    run_eval(step_count=step)
                    eval_idx += 1

            #### FINAL EVALUATION ####
            # This is now outside the loop, guaranteeing it runs when training ends.
            run_eval(step_count=pbar.n)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def save_model(self, e):
        ### save checkpoint
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

            # save the best model when performance_score (λ/C) improves
            if np.mean(self.last_perf_score) > self.last_max_perf_score:
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_max_perf_score = np.mean(self.last_perf_score)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
