import time
from datetime import date
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from utils.misc import temp_seed

today = date.today()


class OnlineSampler:
    def __init__(
        self,
        state_dim: tuple,
        u_dim: int,
        episode_len: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        """
        This computes the ""very"" appropriate parameter for the Monte-Carlo sampling
        given the number of episodes and the given number of cores the runner specified.
        ---------------------------------------------------------------------------------
        Rounds: This gives several rounds when the given sampling load exceeds the number of threads
        the task is assigned. 
        This assigned appropriate parameters assuming one worker work with 2 trajectories.
        """

        # dimensional params
        self.state_dim = state_dim
        self.u_dim = u_dim

        # sampling params
        self.episode_len = episode_len
        self.batch_size = batch_size

        self.episodes_per_worker = 3  # 3 episodes per worker for efficiency
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.total_num_worker = ceil(self.batch_size / (self.thread_batch_size))

        # enforce one thread for each worker to avoid CPU overscription.
        torch.set_num_threads(1)

    def get_reset_data(self, batch_size):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        np.nan makes it easy to debug
        """
        data = dict(
            states=np.full(((batch_size, self.state_dim)), np.nan, dtype=np.float32),
            next_states=np.full(
                ((batch_size, self.state_dim)), np.nan, dtype=np.float32
            ),
            controls=np.full((batch_size, self.u_dim), np.nan, dtype=np.float32),
            rewards=np.full((batch_size, 1), np.nan, dtype=np.float32),
            terminations=np.full((batch_size, 1), np.nan, dtype=np.float32),
            truncations=np.full((batch_size, 1), np.nan, dtype=np.float32),
            logprobs=np.full((batch_size, 1), np.nan, dtype=np.float32),
            entropys=np.full((batch_size, 1), np.nan, dtype=np.float32),
        )
        return data

    def collect_samples(self, env, policy, seed: int | None = None):
        """
        Collect samples in parallel using multiprocessing.
        Appends samples across retries until the half-batch threshold is met.
        """
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))

        policy.to_device(torch.device("cpu"))

        # --- APPEND STRATEGY CONFIG ---
        max_retries = 25
        all_collected_batches = []  # Accumulator for valid worker results
        total_samples_so_far = 0

        with mp.Manager() as manager:
            # Create a fresh queue that definitely works
            queue = manager.Queue()

            for attempt in range(max_retries):
                processes = []
                worker_memories = [None] * self.total_num_worker

                # 1. Spawn Workers
                # If seed is provided, shift it per attempt to avoid duplicate trajectories
                current_seed = seed + (attempt * 1000) if seed is not None else None

                for i in range(self.total_num_worker):
                    args = (i, queue, env, policy, current_seed)
                    p = mp.Process(target=self.collect_trajectory, args=args)
                    processes.append(p)
                    p.start()

                # 2. Collect Results
                expected = len(processes)
                collected = 0

                start_wait = time.time()
                while collected < expected:
                    if time.time() - start_wait > 300:
                        print(
                            f"[Warning] Global collection timeout on attempt {attempt+1}"
                        )
                        break

                    try:
                        pid, data = queue.get(timeout=5.0)
                        if worker_memories[pid] is None:
                            worker_memories[pid] = data
                            collected += 1
                    except Empty:
                        continue

                # 3. Clean up processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    p.join()

                # Filter out failed workers (None)
                valid_this_round = [wm for wm in worker_memories if wm is not None]

                # Add to our accumulator
                all_collected_batches.extend(valid_this_round)

                # Update count
                samples_this_round = sum(len(wm["states"]) for wm in valid_this_round)
                total_samples_so_far += samples_this_round

                # Check threshold (Target: > 80% of batch_size)
                if total_samples_so_far >= (0.8 * self.batch_size):
                    break

            else:
                # Loop finished without break => Failed to reach threshold
                raise RuntimeError(
                    f"Failed to collect sufficient samples. "
                    f"Total collected: {total_samples_so_far} (Threshold: {0.8*self.batch_size})"
                )

            memory = {}
            for wm in all_collected_batches:
                for key, val in wm.items():
                    if key in memory:
                        memory[key] = np.concatenate((memory[key], wm[key]), axis=0)
                    else:
                        memory[key] = wm[key]

        t_end = time.time()
        policy.to_device(device)

        return memory, t_end - t_start

    def collect_trajectory(
        self, pid, queue, env, policy: nn.Module, seed: int | None = None
    ):
        # estimate the batch size to hava a large batch
        data = self.get_reset_data(
            batch_size=self.thread_batch_size + self.episode_len
        )  # allocate memory
        seed = temp_seed(seed, pid)

        current_step = 0
        for i in range(self.episodes_per_worker):
            # env initialization
            obs, _ = env.reset(seed=seed + i)

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, metaData = policy(obs)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)
                    done = term or trunc

                # saving the data
                data["states"][current_step + t] = obs
                data["next_states"][current_step + t] = next_obs
                data["controls"][current_step + t] = a
                data["rewards"][current_step + t] = rew
                data["terminations"][current_step + t] = term
                data["truncations"][current_step + t] = trunc
                data["logprobs"][current_step + t] = (
                    metaData["logprobs"].detach().numpy()
                )
                data["entropys"][current_step + t] = (
                    metaData["entropy"].detach().numpy()
                )

                if done:
                    # clear log
                    current_step += t + 1
                    break

                obs = next_obs

        for k in data:
            data[k] = data[k][:current_step]

        return queue.put([pid, data]) if queue is not None else data
