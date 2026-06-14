import json

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_dim: tuple,
        u_dim: int,
        buffer_size: int,
        batch_size: int,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.control = np.zeros((buffer_size, u_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.termination = np.zeros((buffer_size, 1), dtype=np.float32)

        self.dtype = dtype
        self.device = device

    def pre_process(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def direct_append(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        batch_size = len(states)

        # Create indices for the batch, handling the circular buffer wrap-around
        idxs = (np.arange(batch_size) + self.ptr) % self.buffer_size

        # Update buffer using batch indexing
        self.state[idxs] = self.pre_process(states)
        self.control[idxs] = self.pre_process(controls)
        self.next_state[idxs] = self.pre_process(next_states)
        self.reward[idxs] = self.pre_process(rewards)
        self.termination[idxs] = self.pre_process(terminations)

        # Update pointer and size
        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def random_append(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        batch_size = len(states)

        # 1. Calculate how much space is left to fill sequentially
        free_space = self.buffer_size - self.size
        n_fill = min(batch_size, free_space)

        # 2. FILL PHASE: Append data sequentially into empty slots
        if n_fill > 0:
            # Indices for the sequential part
            fill_idxs = np.arange(self.ptr, self.ptr + n_fill)

            # Slice the batch for the fill part
            self.state[fill_idxs] = self.pre_process(states[:n_fill])
            self.control[fill_idxs] = self.pre_process(controls[:n_fill])
            self.next_state[fill_idxs] = self.pre_process(next_states[:n_fill])
            self.reward[fill_idxs] = self.pre_process(rewards[:n_fill])
            self.termination[fill_idxs] = self.pre_process(terminations[:n_fill])

            # Update pointer and size
            self.ptr = (self.ptr + n_fill) % self.buffer_size
            self.size += n_fill

        # 3. RESERVOIR PHASE: If there is remaining data, overwrite random indices
        n_remain = batch_size - n_fill
        if n_remain > 0:
            # Generate random indices for the overflow data
            rand_idxs = np.random.randint(0, self.buffer_size, size=n_remain)

            # Slice the batch for the remaining part
            self.state[rand_idxs] = self.pre_process(states[n_fill:])
            self.control[rand_idxs] = self.pre_process(controls[n_fill:])
            self.next_state[rand_idxs] = self.pre_process(next_states[n_fill:])
            self.reward[rand_idxs] = self.pre_process(rewards[n_fill:])
            self.termination[rand_idxs] = self.pre_process(terminations[n_fill:])

            # Note: We do not update self.ptr or self.size here because
            # the buffer is already full (size=max) and random overwrite
            # doesn't respect the circular pointer.

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.from_numpy(self.state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.control[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.next_state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.reward[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.termination[ind]).to(self.device).to(self.dtype),
        )

    def save_to_json(self, filepath):
        # Convert numpy arrays to lists for JSON serialization
        buffer_dict = {
            "state": self.state[: self.size].tolist(),
            "control": self.control[: self.size].tolist(),
            "next_state": self.next_state[: self.size].tolist(),
            "reward": self.reward[: self.size].tolist(),
            "termination": self.termination[: self.size].tolist(),
        }

        with open(filepath, "w") as f:
            json.dump(buffer_dict, f)
