import os
import time
from abc import abstractmethod
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base

COLORS = {
    "0": "#4e79a7",  # Blue
    "1": "#f28e2c",  # Orange
    "2": "#e15759",  # Red
    "3": "#76b7b2",  # Teal
    "4": "#59a14f",  # Green
    "5": "#edc949",  # Yellow
    "6": "#af7aa1",  # Purple
    "7": "#ff9da7",  # Pink
    "8": "#9c755f",  # Brown
    "9": "#bab0ab",  # Grey
}


class Evaluator:
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        epochs: int = 10000,
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
        rendering: bool = False,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.env.policy = policy
        self.eval_env.policy = policy
        self.logger = logger
        self.writer = writer
        self.rendering = rendering

        # training parameters
        self.init_epochs = init_epochs
        self.epochs = epochs

        self.log_interval = log_interval
        self.eval_interval = int(epochs / self.log_interval)

        # initialize the essential training components
        self.last_min_auc_mean = 1e10

        self.eval_num = eval_num
        self.eval_episodes = eval_episodes
        self.seed = seed

    @abstractmethod
    def train(self) -> dict[str, float]:
        pass

    def begin_evaluate(self) -> dict[str, float]:
        """For LQR and SD-LQR.

        Returns:
            dict[str, float]: _description_
        """
        start_time = time.time()

        # Train loop
        eval_idx = 1
        self.policy.eval()
        with tqdm(
            initial=self.init_epochs,
            total=(self.init_epochs + self.epochs),
            desc=f"{self.policy.name} Evaluation",
        ) as pbar:
            while pbar.n < (self.init_epochs + self.epochs):
                logging_step = int(self.eval_interval * eval_idx)

                eval_dict_list = []
                for i in range(self.eval_num):
                    eval_dict, supp_dict = self.evaluate()
                    eval_dict_list.append(eval_dict)

                eval_dict = self.average_dict_values(eval_dict_list)

                # Manual logging
                self.write_log(eval_dict, step=logging_step, eval_log=True)
                self.write_image(
                    supp_dict,
                    step=logging_step,
                )

                # Calculate expected remaining time
                pbar.update(self.eval_interval)
                print("pbar updated")
                eval_idx += 1

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def evaluate(self):
        """
        Given one ref, show tracking performance
        """
        dimension = self.eval_env.pos_dimension
        ep_buffers = []
        video_frames = []

        # find mean and CI of data with tqdm that disappears afterward
        for i in tqdm(range(self.eval_num), desc="Evaluating", leave=False):
            track_traj, ref_traj, error_traj, ep_buffer = [], [], [], []
            obs, infos = self.eval_env.reset(seed=self.seed)
            # find mean of data
            for j in range(self.eval_episodes):
                # Env initialization
                options = None if j == 0 else {"replace_x_0": True, "eval_mode": True}
                obs, infos = self.eval_env.reset(seed=self.seed, options=options)

                # Episode variables
                ep_return, ep_ctrl_effort, ep_inf_time = 0, 0, 0
                ep_track_traj, ep_error_traj = [], []

                # Episode rollout
                for t in range(1, self.eval_env.episode_len + 1):
                    if self.rendering and i == 0 and j == 0:
                        frame = self.eval_env.render(mode="rgb_array")
                        if frame is not None:
                            video_frames.append(frame)

                    with torch.no_grad():
                        t0 = time.time()
                        a, _ = self.policy(obs)
                        t1 = time.time()
                        a = (
                            a.cpu().numpy().squeeze(0)
                            if a.shape[-1] > 1
                            else [a.item()]
                        )

                    obs, rew, term, trunc, infos = self.eval_env.step(a)
                    done = term or trunc

                    # === Logging === #
                    gamma = getattr(self.policy, "gamma", 1.0)
                    ep_return += gamma**t * rew
                    ep_ctrl_effort += infos["control_effort"]
                    ep_inf_time += t1 - t0

                    ep_track_traj.append(infos["x"][:dimension])
                    ep_error_traj.append(infos["relative_tracking_error"])
                    if j == 0:
                        ref_traj.append(self.eval_env.xref[t, :dimension])

                    # === Termination logic === #
                    if done:
                        auc = np.trapezoid(ep_error_traj, dx=self.eval_env.dt)

                        ep_buffer.append(
                            {
                                "return": ep_return,
                                "avg_ctrl_effort": ep_ctrl_effort / t,
                                "avg_inf_time": ep_inf_time / t,
                                "mauc": auc * (self.eval_env.episode_len / t),
                                "m2auc": auc * ((self.eval_env.episode_len / t) ** 2),
                                "episode_len": t + 1,
                            }
                        )
                        track_traj.append(ep_track_traj)
                        error_traj.append(ep_error_traj)

                        break

            # === ref traj level logging === #
            rew_list = [ep_info["return"] for ep_info in ep_buffer]
            ctr_list = [ep_info["avg_ctrl_effort"] for ep_info in ep_buffer]
            mauc_list = [ep_info["mauc"] for ep_info in ep_buffer]

            ret_mean, _ = self.mean_confidence_interval(rew_list)
            mauc_mean, _ = self.mean_confidence_interval(mauc_list)
            ctrl_mean, _ = self.mean_confidence_interval(ctr_list)

            C, lbd = self.compute_contraction_rate(error_traj)
            gamma = getattr(self.policy, "gamma", 1.0)
            N_mean, C_gamma_N_mean, C_gamma_inf_mean = self.compute_gamma_discounted_cost(error_traj, gamma, lbd)

            if i == 0:
                fig = self.plot_trajectories(track_traj, error_traj, dimension, C, lbd)

            #
            ep_buffers.append(
                {
                    "return": ret_mean,
                    "avg_ctrl_effort": ctrl_mean,
                    "mauc": mauc_mean,
                    "overshoot": C,
                    "contraction_rate": lbd,
                    "N": N_mean,
                    "C_gamma_N": C_gamma_N_mean,
                    "C_gamma_inf": C_gamma_inf_mean,
                }
            )

        # === eval num level logging === #
        rew_list = [ep_info["return"] for ep_info in ep_buffers]
        ctr_list = [ep_info["avg_ctrl_effort"] for ep_info in ep_buffers]
        mauc_list = [ep_info["mauc"] for ep_info in ep_buffers]
        overshoot_list = [ep_info["overshoot"] for ep_info in ep_buffers]
        lbd_list = [ep_info["contraction_rate"] for ep_info in ep_buffers]
        N_list = [ep_info["N"] for ep_info in ep_buffers]
        C_gamma_N_list = [ep_info["C_gamma_N"] for ep_info in ep_buffers]
        C_gamma_inf_list = [ep_info["C_gamma_inf"] for ep_info in ep_buffers]

        ret_mean, _ = self.mean_confidence_interval(rew_list)
        mauc_mean, _ = self.mean_confidence_interval(mauc_list)
        ctrl_mean, _ = self.mean_confidence_interval(ctr_list)
        overshoot_mean, _ = self.mean_confidence_interval(overshoot_list)
        lbd_mean, _ = self.mean_confidence_interval(lbd_list)
        N_mean_total, _ = self.mean_confidence_interval(N_list)
        C_gamma_N_mean_total, _ = self.mean_confidence_interval(C_gamma_N_list)
        C_gamma_inf_mean_total, _ = self.mean_confidence_interval(C_gamma_inf_list)

        eval_dict = {
            "eval/return": ret_mean,
            "eval/mauc": mauc_mean,
            "eval/control_effort": ctrl_mean,
            "eval/overshoot": overshoot_mean,
            "eval/contraction_rate": lbd_mean,
            "eval/N": N_mean_total,
            "eval/C_gamma_inf": C_gamma_inf_mean_total,
            "eval/performance_score": lbd_mean / (overshoot_mean + 1e-8),
        }
        
        policy_name = self.policy.__class__.__name__.lower() if hasattr(self, "policy") else ""
        if policy_name == "c3m":
            provided_lbd = getattr(self.eval_env, "lbd", getattr(self.policy, "lbd", 0.0))
            eval_dict["eval/contraction_rate_diff"] = lbd_mean - provided_lbd


        if not np.isnan(C_gamma_N_mean_total):
            eval_dict["eval/C_gamma_N"] = C_gamma_N_mean_total

        supp_dict = {f"eval/path_tracking_result": fig}
        if self.rendering and len(video_frames) > 0:
            supp_dict["eval/video"] = np.array(video_frames)

        return eval_dict, supp_dict

    def compute_contraction_rate(self, error_trajectories: list[np.ndarray]):
        """
        Approximates C and lambda such that x(t) <= C * exp(-lambda * t)
        and the AUC (C / lambda) is minimized.
        """

        best_C = 1.0
        best_lbd = 0.0
        min_auc = float("inf")

        # Pre-calculate global max error to determine the search lower bound
        # C must be at least the max error of ANY trajectory to bound it.
        global_max_err = max([np.max(traj) for traj in error_trajectories])
        start_C = max(1.0, global_max_err)

        # Search range for C: From the peak error up to e.g., 10x the peak error
        # We test different "heights" for the envelope.
        c_candidates = np.linspace(start_C, start_C * 10.0, num=100)

        for C_test in c_candidates:
            # 1. Calculate the TIGHTEST lambda for this specific C_test
            # The lambda must satisfy the bound for ALL points in ALL trajectories.
            # constraint: lambda <= (ln(C) - ln(x)) / t

            current_lbd = float("inf")

            valid_C = True
            for err in error_trajectories:
                for i, xe in enumerate(err):
                    t = self.eval_env.dt * (i + 1)  # Avoid divide by zero at t=0
                    if xe <= 1e-6:  # Avoid log(0)
                        continue

                    # Check if this C is physically possible (must start above x)
                    if xe > C_test:
                        valid_C = False
                        break

                    val = (np.log(C_test) - np.log(xe)) / t
                    current_lbd = min(current_lbd, val)

                if not valid_C:
                    break

            if not valid_C or current_lbd <= 0:
                continue

            # 2. Check AUC (Objective Function)
            auc = C_test / current_lbd

            if auc < min_auc:
                min_auc = auc
                best_C = C_test
                best_lbd = current_lbd

        # If minimization fails (e.g. data is weird), fallback to peak
        if best_lbd == 0.0:
            best_C = start_C
            best_lbd = 0.0  # No convergence found

        return best_C, best_lbd

    def compute_gamma_discounted_cost(self, error_trajectories: list[list[float]], gamma: float, lbd: float):
        N_list = []
        c_gamma_N_list = []
        c_gamma_inf_list = []
        dt = self.eval_env.dt
        
        # Determine theoretical overshoot bound factor
        factor = 1.0
        policy_name = self.policy.__class__.__name__.lower() if hasattr(self, "policy") else ""
        if policy_name == "c3m":
            if hasattr(self.policy, "W"):
                W = self.policy.W.detach().cpu().numpy()
                if W.ndim == 3: W = W[0]
                M = W.T @ W
                eigvals = np.linalg.eigvals(M)
                if np.min(np.real(eigvals)) > 0:
                    factor = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
            elif hasattr(self.policy, "CMG") and hasattr(self.eval_env, "x_0"):
                import torch
                x0 = torch.tensor(self.eval_env.x_0, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                W = self.policy.CMG(x0)[0].detach().squeeze(0).cpu().numpy()
                M = W.T @ W
                eigvals = np.linalg.eigvals(M)
                if np.min(np.real(eigvals)) > 0:
                    factor = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
        elif policy_name in ["carl", "trpo", "cpo", "ppo", "cac"]:
            if hasattr(self.policy, "CMG") and hasattr(self.eval_env, "x_0"):
                import torch
                x0 = torch.tensor(self.eval_env.x_0, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                W = self.policy.CMG(x0)[0].detach().squeeze(0).cpu().numpy()
                M = W.T @ W
                eigvals = np.linalg.eigvals(M)
                if np.min(np.real(eigvals)) > 0:
                    cond = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
                    factor = cond / (1 - gamma) if gamma < 1.0 else cond

        for ep_err in error_trajectories:
            N = 0
            for k in range(len(ep_err) - 1, -1, -1):
                # Transient overshoot if cost goes beyond the theoretical bound
                if ep_err[k] > factor * np.exp(-lbd * k * dt):
                    N = k
                    break
            
            # Helper to compute C_gamma for a single trajectory
            def _calc(start, end):
                if start > end: return 0.0
                num = 0.0
                den = 0.0
                for k in range(start, end + 1):
                    if k < len(ep_err):
                        num += (gamma ** k) * ep_err[k]
                        den += (gamma ** k)
                return num / den if den > 0 else 0.0
            
            if N > 1:
                # Transient overshoot detected
                c_gamma_N_list.append(_calc(0, N - 1))
                c_gamma_inf_list.append(_calc(N, len(ep_err) - 1))
            else:
                # No transient overshoot detected (nominal contraction)
                c_gamma_N_list.append(np.nan)
                c_gamma_inf_list.append(_calc(0, len(ep_err) - 1))
                N = 0
                
            N_list.append(N)
            
        return np.nanmean(N_list), np.nanmean(c_gamma_N_list), np.nanmean(c_gamma_inf_list)

    def mean_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        data = np.array(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)  # standard error
        h = 1.96 * sem  # margin of error for 95% CI
        return mean, h

    def plot_trajectories(
        self,
        trajectories: list[np.ndarray],
        error_trajectories: list[np.ndarray],
        dimension: int,
        C: float,
        lbd: float,
    ):
        assert dimension in [1, 2, 3], "Dimension must be 1, 2, or 3."

        # Set subplot parameters based on dimension
        if dimension == 3:
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)  # 2D subplot
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        if dimension in [2, 3]:
            # Dynamically create the coordinate list and plot the reference trajectory
            coords = [self.eval_env.xref[:, i] for i in range(dimension)]
        elif dimension == 1:
            # for one dimensional env (e.g., Segway) we plot x vs time
            coords = [np.arange(len(self.eval_env.xref)), self.eval_env.xref[:, 0]]

        first_point = [c[0] for c in coords]
        ax1.scatter(
            *first_point,
            marker="*",
            # alpha=0.7,
            c="black",
            s=80.0,
        )
        ax1.plot(*coords, linewidth=2.0, linestyle="--", c="black", label="Reference")

        for num_episodes, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            if dimension in [2, 3]:
                coords = [trajectory[:, i] for i in range(dimension)]
            else:
                coords = [np.arange(len(trajectory)), trajectory[:, 0]]
            first_point = [c[0] for c in coords]
            ax1.scatter(
                *first_point,
                marker="*",
                alpha=0.9,
                c=COLORS[str(num_episodes)],
                s=80.0,
            )
            ax1.plot(
                *coords,
                linestyle="-",
                alpha=0.9,
                c=COLORS[str(num_episodes)],
                label=str(num_episodes),
            )

        # Optional: Add axis labels
        if dimension in [2, 3]:
            ax1.set_xlabel("X", fontsize=16)
            ax1.set_ylabel("Y", fontsize=16)
            if dimension == 3:
                ax1.set_zlabel("Z", fontsize=16)
                # Set a nice viewing angle for 3D
                ax1.view_init(elev=25, azim=45)
        else:
            ax1.set_xlabel("Time Steps", fontsize=16)
            ax1.set_ylabel("Position", fontsize=16)

        ax1.set_title("Path Tracking Results", fontsize=18)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # calculate the mean and std of the traj norm error to make plot
        i = 0
        timesteps = np.array(range(self.eval_env.episode_len)) * self.eval_env.dt
        for traj in error_trajectories:
            ax2.plot(
                timesteps[: len(traj)],
                traj,
                c=COLORS[str(i)],
            )
            i += 1
        # plot baseline exponential decay curves
        ax2.plot(timesteps, C * np.exp(-lbd * timesteps), linestyle="--", c="black")
        ax2.set_xlabel("Time (s)", fontsize=16)
        ax2.set_ylabel(r"$||x(t)-x^*(t)||_2 / ||x(0) - x^*(0)||_2$", fontsize=16)

        ax2.set_title(
            rf"Normalized Tracking Error (C={C:.2f}, $\lambda$={lbd:.2f})",
            fontsize=18,
        )
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.close()

        return fig

    @abstractmethod
    def save_model(self, e):
        pass

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, supp_dict: dict, step: int):
        # supp_dict contains fig of plt or video frames
        for key, value in supp_dict.items():
            if "video" in key:
                self.logger.write_videos(step=step, images=value, logdir=key)
            else:
                self.logger.write_images(step=step, image=value, logdir=key)
                import matplotlib.pyplot as plt
                if isinstance(value, plt.Figure):
                    plt.close(value)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict
