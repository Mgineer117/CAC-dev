import time
from abc import abstractmethod

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
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
        self.last_max_perf_score = 0.0

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
                eval_idx += 1

            torch.cuda.empty_cache()

        self.logger.print(
            "total evaluation time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def _compute_empirical_metric_bounds(self, num_samples: int = 2000):
        """Empirical min/max eigenvalues of M(x) = W(x)^{-1} over sampled buffer states.

        This gives a practical (tighter) condition number than the design bounds
        w_ub/w_lb, because the learned metric typically never reaches the extremes
        of those bounds.  The estimate is a lower bound on the true max eigenvalue
        (finite samples miss corners of X), but is far more informative than the
        design bound in practice.

        Returns (empirical_min_eig, empirical_max_eig, empirical_cond), or
        (None, None, None) if the policy has no CMG.
        """
        if not (hasattr(self.policy, "CMG") and hasattr(self.policy, "_bound_W")):
            return None, None, None

        data_x = self.policy.data["x"]
        n = min(num_samples, len(data_x))
        idx = np.random.choice(len(data_x), size=n, replace=False)

        with torch.no_grad():
            x_batch = self.policy.to_tensor(data_x[idx])
            raw_W, _ = self.policy.CMG(x_batch)
            W = self.policy._bound_W(raw_W)
            I = torch.eye(
                self.policy.x_dim,
                device=self.policy.device,
                dtype=self.policy._dtype,
            )
            M = torch.linalg.solve(W, I.unsqueeze(0).expand(n, -1, -1))
            eigs = torch.linalg.eigvalsh(M.cpu()).numpy()  # (n, x_dim), ascending

        empirical_min_eig = float(eigs[:, 0].min())
        empirical_max_eig = float(eigs[:, -1].max())
        empirical_cond = empirical_max_eig / max(empirical_min_eig, 1e-12)
        return empirical_min_eig, empirical_max_eig, empirical_cond

    def evaluate(self):
        """
        Given one ref, show tracking performance.

        Logged metrics (6 total):
          1. eval/auc              — normalised AUC of tracking error
          2. eval/control_effort   — mean squared control magnitude
          3. eval/contraction_rate — fitted λ of the exponential envelope
          4. eval/latency_ms       — policy inference latency in ms
          5. eval/performance_score — λ / C  (contraction efficiency)
          6. eval/contraction_flag  — 1 if empirical C exceeds theory bound
        """
        dimension = self.eval_env.pos_dimension
        ep_buffers = []
        video_frames = []

        policy_name = (
            self.policy.__class__.__name__.lower()
            if hasattr(self, "policy")
            else ""
        )

        # Practical eigenvalue bounds — computed once per evaluate() call over
        # buffer states.  Tighter than the design bounds w_ub/w_lb but not
        # certified (finite sample); treated as an empirical lower/upper bound.
        emp_min_eig, emp_max_eig, emp_cond = self._compute_empirical_metric_bounds()
        # Make empirical cond available to env.render() without an extra CMG call.
        self.eval_env.emp_cond = emp_cond

        # find mean and CI of data with tqdm that disappears afterward
        for i in tqdm(range(self.eval_num), desc="Evaluating", leave=False):
            track_traj, ref_traj, error_traj, ep_buffer = [], [], [], []
            for j in range(self.eval_episodes):
                # Env initialization
                options = None if j == 0 else {"replace_x_0": True, "eval_mode": True}
                obs, infos = self.eval_env.reset(seed=self.seed, options=options)

                # Episode variables
                ep_ctrl_effort, ep_inf_time = 0, 0
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
                                "u_norm": ep_ctrl_effort / t,
                                "avg_inf_time":    ep_inf_time / t,
                                "mauc":            auc * (self.eval_env.episode_len / t),
                                # ‖e(0)‖₂ — used for the contraction-flag bound
                                "init_cost":       np.sqrt(
                                    max(self.eval_env.init_tracking_error, 0.0)
                                ),
                            }
                        )
                        track_traj.append(ep_track_traj)
                        error_traj.append(ep_error_traj)

                        break

            # === ref traj level logging === #
            ctr_list  = [ep["u_norm"] for ep in ep_buffer]
            mauc_list = [ep["mauc"]            for ep in ep_buffer]
            inf_list  = [ep["avg_inf_time"]    for ep in ep_buffer]
            init_cost_list = [ep["init_cost"]  for ep in ep_buffer]

            mauc_mean, _ = self.mean_confidence_interval(mauc_list)
            ctrl_mean, _ = self.mean_confidence_interval(ctr_list)
            inf_mean, _  = self.mean_confidence_interval(inf_list)
            init_cost_mean = float(np.mean(init_cost_list))

            C, lbd = self.compute_contraction_rate(error_traj)

            if i == 0:
                fig = self.plot_trajectories(
                    track_traj, error_traj, dimension, C, lbd,
                    emp_cond=emp_cond,
                )

            ep_buffers.append(
                {
                    "u_norm": ctrl_mean,
                    "mauc":            mauc_mean,
                    "avg_inf_time":    inf_mean,
                    "overshoot":       C,
                    "contraction_rate": lbd,
                    "init_cost":       init_cost_mean,
                }
            )

        # === eval num level logging === #
        ctr_list       = [ep["u_norm"]  for ep in ep_buffers]
        mauc_list      = [ep["mauc"]              for ep in ep_buffers]
        inf_list       = [ep["avg_inf_time"]      for ep in ep_buffers]
        overshoot_list = [ep["overshoot"]         for ep in ep_buffers]
        lbd_list       = [ep["contraction_rate"]  for ep in ep_buffers]
        init_cost_list = [ep["init_cost"]         for ep in ep_buffers]

        mauc_mean, _        = self.mean_confidence_interval(mauc_list)
        ctrl_mean, _        = self.mean_confidence_interval(ctr_list)
        inf_mean_total, _   = self.mean_confidence_interval(inf_list)
        overshoot_mean, _   = self.mean_confidence_interval(overshoot_list)
        lbd_mean, _         = self.mean_confidence_interval(lbd_list)
        init_cost_total     = float(np.mean(init_cost_list))

        # --- Contraction flags --------------------------------------------------
        gamma      = getattr(self.policy, "gamma", 1.0)
        w_ub       = float(getattr(self.policy, "w_ub", 1.0))
        w_lb       = float(getattr(self.policy, "w_lb", 1.0))
        sqrt_cond  = np.sqrt(w_ub / (w_lb + 1e-12))

        if policy_name == "c3m":
            scale = 1.0
        else:
            scale = 1.0 / max(1.0 - gamma, 1e-8)

        # Theoretical bound (design bounds w_ub / w_lb — conservative)
        theo_bound = scale * sqrt_cond * init_cost_total
        contraction_flag = float(overshoot_mean > theo_bound)

        # Practical bound (empirical eigenvalues over buffer — tighter but not
        # certified; finite samples may miss extremes of X)
        if emp_cond is not None:
            sqrt_emp_cond = np.sqrt(emp_cond)
            practical_bound = scale * sqrt_emp_cond * init_cost_total
            practical_contraction_flag = float(overshoot_mean > practical_bound)
        else:
            practical_contraction_flag = float("nan")

        eval_dict = {
            "eval/auc":                       mauc_mean,
            "eval/u_norm":                    ctrl_mean,
            "eval/lambda":                    lbd_mean,
            "eval/overshoot":                 overshoot_mean,
            "eval/latency_ms":                inf_mean_total * 1e3,
            "eval/performance_score":         lbd_mean / (overshoot_mean + 1e-8),
            "eval/contraction_flag":          contraction_flag,
            "eval/practical_contraction_flag": practical_contraction_flag,
        }
        if emp_min_eig is not None:
            eval_dict["eval/empirical_min_eig"] = emp_min_eig
            eval_dict["eval/empirical_max_eig"] = emp_max_eig
            eval_dict["eval/empirical_cond"]    = emp_cond

        supp_dict = {"eval/path_tracking_result": fig}
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
        emp_cond: float | None = None,
    ):
        """Plot path tracking results and normalised tracking error.

        The right panel (ax2) shows:
          - each episode's normalised error trajectory
          - the fitted exponential envelope  C · exp(−λ t)  (black dashed)
          - the theoretical upper bound (red/blue dashed) from contraction theory
          - for CARL_M: the accelerated IES bound from Theorem (Accelerated IES),
            using the running-average contraction rate λ̄(0,k).
        """
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

        # ── Right panel: normalised tracking error + theory bound ──────────────
        timesteps = np.array(range(self.eval_env.episode_len)) * self.eval_env.dt

        for i, traj in enumerate(error_trajectories):
            ax2.plot(
                timesteps[: len(traj)],
                traj,
                c=COLORS[str(i)],
                alpha=0.8,
            )

        # Fitted empirical envelope: C · exp(−λ t)
        ax2.plot(
            timesteps,
            C * np.exp(-lbd * timesteps),
            linestyle="--",
            c="black",
            linewidth=1.5,
            label=rf"Fitted: $C={C:.2f}$, $\lambda={lbd:.2f}$",
        )

        # Theoretical upper bound from contraction theory
        policy_name = (
            self.policy.__class__.__name__.lower()
            if hasattr(self, "policy")
            else ""
        )
        gamma      = getattr(self.policy, "gamma", 1.0)
        w_ub       = float(getattr(self.policy, "w_ub", None) or 1.0)
        w_lb       = float(getattr(self.policy, "w_lb", None) or 1.0)
        lbd_design = float(getattr(self.policy, "lbd", lbd))
        cond       = w_ub / (w_lb + 1e-12)

        if policy_name == "c3m":
            theo_factor = np.sqrt(cond)
            bound_label = rf"Theory: $\sqrt{{w_{{ub}}/w_{{lb}}}}\,e^{{-\lambda t}}$"
        else:
            theo_factor = (1.0 / max(1.0 - gamma, 1e-8)) * np.sqrt(cond)
            bound_label = rf"Theory: $\frac{{1}}{{1-\gamma}}\sqrt{{w_{{ub}}/w_{{lb}}}}\,e^{{-\lambda t}}$"

        ax2.plot(
            timesteps,
            theo_factor * np.exp(-lbd_design * timesteps),
            linestyle="-.",
            color="crimson",
            linewidth=1.5,
            label=bound_label,
        )

        # Practical bound — replaces design sqrt(κ) with empirical sqrt(κ̂)
        if emp_cond is not None:
            sqrt_emp_cond = np.sqrt(emp_cond)
            if policy_name == "c3m":
                prac_factor = sqrt_emp_cond
                prac_label = rf"Practical: $\sqrt{{\hat\kappa}}\,e^{{-\lambda t}}$ ($\hat\kappa={emp_cond:.1f}$)"
            else:
                # RL bounds carry the discounted-horizon 1/(1-gamma) factor.
                prac_factor = (1.0 / max(1.0 - gamma, 1e-8)) * sqrt_emp_cond
                prac_label = rf"Practical: $\frac{{1}}{{1-\gamma}}\sqrt{{\hat\kappa}}\,e^{{-\lambda t}}$ ($\hat\kappa={emp_cond:.1f}$)"
            ax2.plot(
                timesteps,
                prac_factor * np.exp(-lbd_design * timesteps),
                linestyle=":",
                color="darkorange",
                linewidth=1.8,
                label=prac_label,
            )

        ax2.set_xlabel("Time (s)", fontsize=16)
        ax2.set_ylabel(r"$\|x(t)-x^*(t)\|_2 / \|x(0)-x^*(0)\|_2$", fontsize=16)
        ax2.set_title(
            rf"Normalised Tracking Error ($\lambda_{{design}}={lbd_design:.2f}$)",
            fontsize=18,
        )
        ax2.legend(fontsize=9)
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
