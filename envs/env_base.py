from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil, floor
from time import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class BaseEnv(gym.Env):
    """Base class for all environments."""

    def __init__(self, env_config: dict):
        super(BaseEnv, self).__init__()

        # X bounds
        self.X_MIN = env_config["x_min"]
        self.X_MAX = env_config["x_max"]

        # Initial reference state bounds
        self.XREF_INIT_MIN = env_config["xref_init_min"]
        self.XREF_INIT_MAX = env_config["xref_init_max"]

        # Initial reference state perturbation bounds
        self.XE_INIT_MIN = env_config["xe_init_min"]
        self.XE_INIT_MAX = env_config["xe_init_max"]

        # Reference state perturbation bounds for c3m
        self.XE_MIN = env_config["xe_min"]
        self.XE_MAX = env_config["xe_max"]

        # Reference control bounds
        self.UREF_MIN = env_config["uref_min"]
        self.UREF_MAX = env_config["uref_max"]

        # environment parameters
        self.num_dim_x = env_config["num_dim_x"]  # x, y, theta, v
        self.num_dim_control = env_config[
            "num_dim_control"
        ]  # u1 (angular acc), u2 (linear acc)
        self.pos_dimension = env_config["pos_dimension"]

        self.time_bound = env_config["time_bound"]
        self.dt = env_config["dt"]
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        # state window
        assert env_config["num_windows"] > 0, "num_windows must be positive."
        if env_config["num_windows"] > self.max_episode_len:
            self.num_windows = self.max_episode_len
        else:
            self.num_windows = env_config["num_windows"]

        # dynamics parameters
        self.tracking_scaler = env_config["q"]
        self.control_scaler = env_config["r"]

        # etc parameters
        self.use_learned_dynamics = False
        self.sample_mode = env_config["sample_mode"]
        self.reward_mode = env_config["reward_mode"]

        # overall state bounds
        ref_unit_min = np.concatenate((self.X_MIN.flatten(), self.UREF_MIN.flatten()))
        ref_unit_max = np.concatenate((self.X_MAX.flatten(), self.UREF_MAX.flatten()))

        self.STATE_MIN = np.concatenate(
            (self.X_MIN.flatten(), np.tile(ref_unit_min, self.num_windows))
        )
        self.STATE_MAX = np.concatenate(
            (self.X_MAX.flatten(), np.tile(ref_unit_max, self.num_windows))
        )

        # gymnasium spaces
        self.observation_space = spaces.Box(
            low=self.STATE_MIN.flatten(),
            high=self.STATE_MAX.flatten(),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=self.UREF_MIN.flatten(),
            high=self.UREF_MAX.flatten(),
            dtype=np.float64,
        )

        # reset
        self.reset()

    def get_horizon_matched_gamma(self, scale: float = 1.0):
        """Scale \in (0, 1] determines how much we prioritize short-term vs long-term rewards."""
        scale = max(1e-3, min(scale, 1.0))
        return round(1 - (1 / (scale * self.max_episode_len)), 3)

    def reset(self, seed=None, options: dict | None = None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.time_steps = 0

        # Initialize the state
        if options is None:
            # Default reset behavior
            self.x_t, self.xref, self.uref, self.episode_len = self.system_reset()
            # Clip reference trajectory to X bounds (all dims).
            # Position dims are already handled by get_transition's freeze logic;
            # this catches non-position dims (velocities, etc.) that may drift OOB.
            self.xref = np.clip(self.xref, self.X_MIN.flatten(), self.X_MAX.flatten())
        else:
            assert hasattr(self, "xref") and hasattr(
                self, "uref"
            ), "Custom reset requires predefined xref and uref."

            # Custom reset behavior that keeps reference trajectory but changes initial state
            if options.get("replace_x_0", True):
                _, xe_0, _ = self.define_initial_state()
                self.x_t = self.xref[0] + xe_0
            else:
                raise NotImplementedError(
                    "Only replace_x_0 option is implemented for now."
                )

        self.x_0 = self.x_t.copy()
        state = self.construct_state(self.x_t)
        self.init_tracking_error = np.linalg.norm(self.x_t - self.xref[0], ord=2) ** 2

        self.traj_x, self.traj_y, self.traj_z = [], [], []
        self.err_history, self.bound_history, self.rt_bound_history = [], [], []

        return state, {"x": self.x_t, "tracking_error": self.init_tracking_error}

    def step(self, u):
        """Run one timestep of the environment's dynamics."""
        # Update time step
        self.time_steps += 1

        # Construct u and apply u clipping
        u = self.uref[self.time_steps] + u
        self.current_u = u.copy()
        # Get reward
        reward, infos = self.get_rewards(u)

        # Get next state
        next_x, next_x_wrapped, termination, truncation, _ = self.get_transition(
            self.x_t, u
        )

        # clip state to bounds
        next_x_wrapped = np.clip(
            next_x_wrapped, self.X_MIN.flatten(), self.X_MAX.flatten()
        )

        # Construct observation
        state = self.construct_state(next_x_wrapped)
        self.x_t = next_x

        return (
            state,
            reward,
            termination,
            truncation,
            {
                "x": next_x_wrapped,
                "tracking_error": infos["tracking_error"],
                "control_effort": infos["control_effort"],
                "relative_tracking_error": infos["tracking_error"]
                / self.init_tracking_error,
            },
        )

    def get_transition(self, x: np.ndarray, u: np.ndarray):
        """Compute the next state given current state x and action u.

        Boundary handling: all state dims transition normally. If the resulting
        next position would exceed X bounds, only the position dims are reverted
        to their previous values — velocities, angles, and all other dims still
        advance. Episodes never terminate due to boundary violations.
        """
        x_dot = self.get_dynamics(x, u)
        next_x = x + self.dt * x_dot

        # Revert only position dims when next position is out-of-bounds.
        pos_min = self.X_MIN.flatten()[: self.pos_dimension]
        pos_max = self.X_MAX.flatten()[: self.pos_dimension]
        next_pos = next_x[: self.pos_dimension]
        if np.any(next_pos < pos_min) or np.any(next_pos > pos_max):
            next_x[: self.pos_dimension] = x[: self.pos_dimension]

        next_x_wrapped = self.wrap_angles(next_x)

        # Episodes only end by time-truncation, never by boundary violation.
        termination = False
        truncation = self.time_steps == self.episode_len - 1

        return next_x, next_x_wrapped, termination, truncation, x_dot

    def get_dynamics(self, x: np.ndarray, u: np.ndarray):
        """Compute the dynamics x_dot given current state x and action u."""
        f_x, B_x, _ = self.get_f_and_B(x)

        # add warning if u is nan
        if np.any(np.isnan(u)):
            print("[Warning]: NaN values found in control input u.")
            u = np.nan_to_num(u)
        x_dot = f_x + np.matmul(B_x, u[..., np.newaxis]).squeeze()

        return x_dot

    def get_f_and_B(self, x: torch.Tensor | np.ndarray):
        """Get f(x), B(x), and B_null(x) using either learned dynamics or analytical functions."""
        if self.use_learned_dynamics:
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(self.wrap_angles(x))
            return (
                f_x.cpu().squeeze(0).numpy(),
                B_x.cpu().squeeze(0).numpy(),
                Bbot_x.cpu().squeeze(0).numpy(),
            )
        else:
            return self.f_func(x), self.B_func(x), self.B_null(x)

    def wrap_angles(self, x: np.ndarray):
        x_copy = x.copy()

        # wrap angles between -pi to pi
        for idx in getattr(self, "angle_idx", []):
            x_copy[idx] = (x_copy[idx] + np.pi) % (2 * np.pi) - np.pi
        return x_copy

    def construct_state(self, x: np.ndarray):
        # 1. Define the slice window normally
        start = self.time_steps
        end = min(self.time_steps + self.num_windows, self.episode_len)

        # 2. Get the available real data
        # (If we are at the end, this will just be shorter than normal)
        x_window = self.xref[start:end]
        u_window = self.uref[start:end]

        # 3. Calculate how much is missing
        pad_len = self.num_windows - len(x_window)

        # 4. Pad if necessary (automatic "if/else" inside numpy)
        if pad_len > 0:
            # Pad the first dimension (rows) with the edge value
            x_window = np.pad(x_window, ((0, pad_len), (0, 0)), mode="edge")
            u_window = np.pad(u_window, ((0, pad_len), (0, 0)), mode="edge")

        return np.concatenate([x, x_window.flatten(), u_window.flatten()])

    def replace_dynamics(self, dynamics_model: nn.Module):
        print("[INFO] The environment is now using learned dynamics for transition.")
        self.learned_dynamics_model = deepcopy(dynamics_model).cpu()
        self.learned_dynamics_model.device = torch.device("cpu")
        self.use_learned_dynamics = True

    @abstractmethod
    def _f_logic(self, x: torch.Tensor | np.ndarray, lib):
        """Logic for calculating f(x) given a library (torch or numpy)."""
        pass

    @abstractmethod
    def _B_logic(self, x: torch.Tensor | np.ndarray, lib):
        """Logic for calculating B(x) given a library (torch or numpy)."""
        pass

    def _B_null_logic(self, x, n, lib):
        """Builds the B_null matrix batch using the provided library."""

        # Calculate the dimensions for the component matrices
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)

        if lib == torch:
            # 1. Create the base 2D matrix
            Bbot = torch.cat(
                (torch.eye(eye_dims), torch.zeros(zero_dims)),
                dim=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            return Bbot.repeat(n, 1, 1)
        else:  # lib == np
            # 1. Create the base 2D matrix
            Bbot = np.concatenate(
                (np.eye(eye_dims), np.zeros(zero_dims)),
                axis=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            #    (np.newaxis adds the first dimension for repeating)
            return np.repeat(Bbot[np.newaxis, :, :], n, axis=0)

    def f_func(self, x: torch.Tensor | np.ndarray):
        """Calculates the drift dynamics f(x) for torch or numpy."""
        if isinstance(x, torch.Tensor):
            lib = torch
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            result = self._f_logic(x, lib)
        else:
            lib = np
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            result = self._f_logic(x, lib)

        try:
            return result.squeeze(0)
        except:
            return result

    def B_func(self, x: torch.Tensor | np.ndarray):
        """Calculates the control matrix B(x) for torch or numpy."""
        if isinstance(x, torch.Tensor):
            lib = torch
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            result = self._B_logic(x, lib)
        else:
            lib = np
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            result = self._B_logic(x, lib)

        try:
            return result.squeeze(0)
        except:
            return result

    def B_null(self, x: torch.Tensor | np.ndarray):
        """Calculates the null space of B for torch or numpy."""

        # Check type and get the batch size 'n'
        if isinstance(x, torch.Tensor):
            lib = torch
            n = 1 if len(x.shape) == 1 else x.shape[0]
            result = self._B_null_logic(x, n, lib)
        else:
            lib = np
            n = 1 if len(x.shape) == 1 else x.shape[0]
            result = self._B_null_logic(x, n, lib)

        # .squeeze() removes the batch dimension if the input was 1D
        try:
            return result.squeeze(0)
        except:
            return result

    def define_initial_state(self):
        """Define the initial state of the environment."""
        xref_0 = self.XREF_INIT_MIN + np.random.rand(len(self.XREF_INIT_MIN)) * (
            self.XREF_INIT_MAX - self.XREF_INIT_MIN
        )
        xe_0 = self.XE_INIT_MIN + np.random.rand(len(self.XE_INIT_MIN)) * (
            self.XE_INIT_MAX - self.XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0

        return xref_0, xe_0, x_0

    @abstractmethod
    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise):
        """Sample reference controls based on frequencies and weights."""
        pass

    @abstractmethod
    def system_reset(self):
        pass

    def render(self, mode="human"):
        if not hasattr(self, "fig"):
            import matplotlib.pyplot as plt
            if mode == "rgb_array":
                plt.switch_backend('Agg')
            elif mode == "human":
                plt.ion()
            self.fig = plt.figure(figsize=(12, 6))
            if self.pos_dimension == 3:
                self.ax = self.fig.add_subplot(121, projection='3d')
            else:
                self.ax = self.fig.add_subplot(121)
            self.ax_err = self.fig.add_subplot(122)
            self.traj_x, self.traj_y, self.traj_z = [], [], []
            self.err_history, self.bound_history, self.rt_bound_history = [], [], []

        self.ax.clear()
        self.ax_err.clear()
        import numpy as np

        # Calculate Contraction Bounds
        bound = None
        policy_name = self.policy.__class__.__name__.lower() if hasattr(self, "policy") else ""
        if policy_name in ["c3m", "lqr", "carl", "trpo", "cpo", "sd_lqr", "ppo", "cac", "algorithm"]:
            lbd = getattr(self, "lbd", getattr(self.policy, "lbd", 0.0))
            gamma = getattr(self, "gamma", getattr(self.policy, "gamma", 1.0))
            
            cond = 1.0
            if policy_name in ["c3m", "lqr", "sd_lqr"]:
                if hasattr(self.policy, "W"):
                    W = self.policy.W.detach().cpu().numpy()
                    if W.ndim == 3: W = W[0]
                    M = W.T @ W
                    eigvals = np.linalg.eigvals(M)
                    if np.min(np.real(eigvals)) > 0:
                        cond = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
                elif hasattr(self.policy, "CMG") and hasattr(self, "x_0"):
                    import torch
                    x0 = torch.tensor(self.x_0, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                    W = self.policy.CMG(x0)[0].detach().squeeze(0).cpu().numpy()
                    M = W.T @ W
                    eigvals = np.linalg.eigvals(M)
                    if np.min(np.real(eigvals)) > 0:
                        cond = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
            elif policy_name in ["carl", "trpo", "cpo", "ppo", "cac"]:
                if hasattr(self.policy, "CMG") and hasattr(self, "x_0"):
                    import torch
                    x0 = torch.tensor(self.x_0, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                    W = self.policy.CMG(x0)[0].detach().squeeze(0).cpu().numpy()
                    M = W.T @ W
                    eigvals = np.linalg.eigvals(M)
                    if np.min(np.real(eigvals)) > 0:
                        cond = np.sqrt(np.max(np.real(eigvals)) / np.min(np.real(eigvals)))
            
            if hasattr(self, "x_0") and hasattr(self, "xref"):
                e0 = self.x_0 - self.xref[0]
                e0 = self.wrap_angles(e0)
                C_s0 = np.linalg.norm(e0)
            else:
                C_s0 = 1.0

            if policy_name in ["c3m"]:
                bound = cond * C_s0 * np.exp(-lbd * self.time_steps * self.dt)
                bound_arr = cond * C_s0 * np.exp(-lbd * np.arange(self.episode_len) * self.dt)
            elif policy_name in ["carl", "trpo", "cpo", "ppo", "cac"]:
                factor = cond / (1 - gamma) if gamma < 1.0 else cond
                bound = factor * C_s0 * np.exp(-lbd * self.time_steps * self.dt)
                bound_arr = factor * C_s0 * np.exp(-lbd * np.arange(self.episode_len) * self.dt)
            elif policy_name in ["lqr", "sd_lqr"]:
                bound = cond * C_s0 * np.exp(-lbd * self.time_steps * self.dt)
                bound_arr = cond * C_s0 * np.exp(-lbd * np.arange(self.episode_len) * self.dt)
                if not hasattr(self, "empirical_alpha"):
                    self.empirical_alpha = 1.0
                
                if hasattr(self, "xref"):
                    current_error = np.linalg.norm(self.wrap_angles(self.x_t - self.xref[self.time_steps]))
                    current_bound = bound if bound > 1e-9 else 1e-9
                    alpha = current_error / current_bound
                    if alpha > self.empirical_alpha:
                        self.empirical_alpha = alpha
        
        # Track histories for error plot
        if hasattr(self, "xref"):
            e_t = self.wrap_angles(self.x_t - self.xref[self.time_steps])
            self.err_history.append(np.linalg.norm(e_t))
        if bound is not None:
            self.bound_history.append(bound)
            if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha"):
                self.rt_bound_history.append(self.empirical_alpha * bound)

        # Extract position and yaw
        pos = self.x_t[:self.pos_dimension]
        yaw = 0.0
        if hasattr(self, "angle_idx") and len(self.angle_idx) > 0:
            yaw = self.x_t[self.angle_idx[-1]]

        # Compute velocity/acceleration vector from dynamics (x_dot)
        if hasattr(self, "current_u"):
            x_dot = self.get_dynamics(self.x_t, self.current_u)
            action_vec = x_dot[:self.pos_dimension]
        else:
            action_vec = np.zeros(self.pos_dimension)

        if self.pos_dimension == 3:
            # Draw reference trajectory
            if hasattr(self, "xref"):
                self.ax.plot(self.xref[:, 0], self.xref[:, 1], self.xref[:, 2], 'k:', label='Reference')

            self.traj_x.append(pos[0])
            self.traj_y.append(pos[1])
            self.traj_z.append(pos[2])
            self.ax.plot(self.traj_x, self.traj_y, self.traj_z, 'b-', label='Trajectory')
            self.ax.scatter(pos[0], pos[1], pos[2], color='r', s=50)

            # Draw yaw direction
            arrow_len = 1.0
            dx, dy = arrow_len * np.cos(yaw), arrow_len * np.sin(yaw)
            self.ax.quiver(pos[0], pos[1], pos[2], dx, dy, 0, color='g', length=1.0, normalize=True, label='Yaw')

            # Draw action command (velocity)
            if np.linalg.norm(action_vec) > 1e-3:
                self.ax.quiver(pos[0], pos[1], pos[2], action_vec[0], action_vec[1], action_vec[2], color='m', length=1.0, normalize=True, label='Action')
                
            if hasattr(self, "xref"):
                margin = max(0.5, bound * 1.1 if bound is not None else 0.0)
                if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha") and bound is not None:
                    margin = max(margin, self.empirical_alpha * bound * 1.1)
                self.ax.set_xlim(np.min(self.xref[:, 0]) - margin, np.max(self.xref[:, 0]) + margin)
                self.ax.set_ylim(np.min(self.xref[:, 1]) - margin, np.max(self.xref[:, 1]) + margin)
                self.ax.set_zlim(np.min(self.xref[:, 2]) - margin, np.max(self.xref[:, 2]) + margin)

                if bound is not None:
                    u_grid, v_grid = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
                    step = max(1, self.episode_len // 20)
                    for t in range(0, self.episode_len, step):
                        ref_pos = self.xref[t, :3]
                        b_t = bound_arr[t]
                        x_sphere = ref_pos[0] + b_t * np.cos(u_grid) * np.sin(v_grid)
                        y_sphere = ref_pos[1] + b_t * np.sin(u_grid) * np.sin(v_grid)
                        z_sphere = ref_pos[2] + b_t * np.cos(v_grid)
                        self.ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="orange", alpha=0.05)
                        
                    self.ax.plot([], [], [], color="orange", alpha=0.5, label='Theoretical Bound')
                    
                    # Highlight current theoretical bound
                    cur_ref_pos = self.xref[self.time_steps, :3]
                    cur_b_t = bound_arr[self.time_steps]
                    cur_x = cur_ref_pos[0] + cur_b_t * np.cos(u_grid) * np.sin(v_grid)
                    cur_y = cur_ref_pos[1] + cur_b_t * np.sin(u_grid) * np.sin(v_grid)
                    cur_z = cur_ref_pos[2] + cur_b_t * np.cos(v_grid)
                    self.ax.plot_wireframe(cur_x, cur_y, cur_z, color="orange", alpha=0.5, label='Current Theo Bound')
                    
                    if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha"):
                        rt_bound_arr = self.empirical_alpha * bound_arr
                        for t in range(0, self.episode_len, step):
                            ref_pos = self.xref[t, :3]
                            rt_b_t = rt_bound_arr[t]
                            x_rt = ref_pos[0] + rt_b_t * np.cos(u_grid) * np.sin(v_grid)
                            y_rt = ref_pos[1] + rt_b_t * np.sin(u_grid) * np.sin(v_grid)
                            z_rt = ref_pos[2] + rt_b_t * np.cos(v_grid)
                            self.ax.plot_wireframe(x_rt, y_rt, z_rt, color="red", alpha=0.05)
                        self.ax.plot([], [], [], color="red", alpha=0.5, label='Real-time Bound')
                        
                        # Highlight current real-time bound
                        cur_rt_b_t = rt_bound_arr[self.time_steps]
                        cur_x_rt = cur_ref_pos[0] + cur_rt_b_t * np.cos(u_grid) * np.sin(v_grid)
                        cur_y_rt = cur_ref_pos[1] + cur_rt_b_t * np.sin(u_grid) * np.sin(v_grid)
                        cur_z_rt = cur_ref_pos[2] + cur_rt_b_t * np.cos(v_grid)
                        self.ax.plot_wireframe(cur_x_rt, cur_y_rt, cur_z_rt, color="red", alpha=0.5, label='Current RT Bound')
            else:
                self.ax.set_xlim(self.X_MIN[0], self.X_MAX[0])
                self.ax.set_ylim(self.X_MIN[1], self.X_MAX[1])
                self.ax.set_zlim(self.X_MIN[2], self.X_MAX[2])

        else:
            # 1D or 2D
            p_x = pos[0]
            p_y = pos[1] if self.pos_dimension > 1 else 0.0
            
            # Draw reference trajectory
            if hasattr(self, "xref"):
                ref_x = self.xref[:, 0]
                ref_y = self.xref[:, 1] if self.pos_dimension > 1 else np.zeros_like(ref_x)
                self.ax.plot(ref_x, ref_y, 'k:', label='Reference')

            self.traj_x.append(p_x)
            self.traj_y.append(p_y)
            self.ax.plot(self.traj_x, self.traj_y, 'b-', label='Trajectory')
            self.ax.scatter(p_x, p_y, color='r', s=50)

            # Draw yaw
            arrow_len = 1.0
            dx, dy = arrow_len * np.cos(yaw), arrow_len * np.sin(yaw)
            if self.pos_dimension > 1:
                self.ax.arrow(p_x, p_y, dx, dy, color='g', head_width=0.2, label='Yaw')

            # Draw action command (velocity)
            if np.linalg.norm(action_vec) > 1e-3:
                ax_dx = action_vec[0]
                ax_dy = action_vec[1] if self.pos_dimension > 1 else 0.0
                self.ax.arrow(p_x, p_y, ax_dx, ax_dy, color='m', head_width=0.2, label='Action')
                
            if hasattr(self, "xref"):
                margin = max(0.5, bound * 1.1 if bound is not None else 0.0)
                if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha") and bound is not None:
                    margin = max(margin, self.empirical_alpha * bound * 1.1)
                self.ax.set_xlim(np.min(self.xref[:, 0]) - margin, np.max(self.xref[:, 0]) + margin)
                if self.pos_dimension > 1:
                    self.ax.set_ylim(np.min(self.xref[:, 1]) - margin, np.max(self.xref[:, 1]) + margin)

                if bound is not None:
                    import matplotlib.pyplot as plt
                    step = max(1, self.episode_len // 50)
                    for t in range(0, self.episode_len, step):
                        ref_pos = self.xref[t, :2]
                        b_t = bound_arr[t]
                        circle = plt.Circle((ref_pos[0], ref_pos[1] if self.pos_dimension > 1 else 0.0), b_t, color='orange', fill=True, alpha=0.03, linewidth=0)
                        self.ax.add_patch(circle)
                        circle_edge = plt.Circle((ref_pos[0], ref_pos[1] if self.pos_dimension > 1 else 0.0), b_t, color='orange', fill=False, alpha=0.2, linestyle=':')
                        self.ax.add_patch(circle_edge)
                        
                    self.ax.plot([], [], color='orange', alpha=0.5, label='Theoretical Bound')
                    
                    # Highlight current theoretical bound
                    cur_ref_pos = self.xref[self.time_steps, :2]
                    cur_b_t = bound_arr[self.time_steps]
                    cur_circle = plt.Circle((cur_ref_pos[0], cur_ref_pos[1] if self.pos_dimension > 1 else 0.0), cur_b_t, color='orange', fill=False, linewidth=2.0, label='Current Theo Bound')
                    self.ax.add_patch(cur_circle)
                    
                    if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha"):
                        rt_bound_arr = self.empirical_alpha * bound_arr
                        for t in range(0, self.episode_len, step):
                            ref_pos = self.xref[t, :2]
                            rt_b_t = rt_bound_arr[t]
                            rt_circle = plt.Circle((ref_pos[0], ref_pos[1] if self.pos_dimension > 1 else 0.0), rt_b_t, color='red', fill=True, alpha=0.03, linewidth=0)
                            self.ax.add_patch(rt_circle)
                            rt_circle_edge = plt.Circle((ref_pos[0], ref_pos[1] if self.pos_dimension > 1 else 0.0), rt_b_t, color='red', fill=False, alpha=0.2, linestyle=':')
                            self.ax.add_patch(rt_circle_edge)
                            
                        self.ax.plot([], [], color='red', alpha=0.5, label='Real-time Bound')
                        
                        # Highlight current real-time bound
                        cur_rt_b_t = rt_bound_arr[self.time_steps]
                        cur_rt_circle = plt.Circle((cur_ref_pos[0], cur_ref_pos[1] if self.pos_dimension > 1 else 0.0), cur_rt_b_t, color='red', fill=False, linewidth=2.0, linestyle=':', label='Current RT Bound')
                        self.ax.add_patch(cur_rt_circle)
            else:
                self.ax.set_xlim(self.X_MIN[0], self.X_MAX[0])
                if self.pos_dimension > 1:
                    self.ax.set_ylim(self.X_MIN[1], self.X_MAX[1])
                
        # Add timestep to title
        if hasattr(self, "time_steps"):
            title = f"Time Step: {self.time_steps}"
            if policy_name in ["lqr", "sd_lqr"] and hasattr(self, "empirical_alpha"):
                title += f" | alpha: {self.empirical_alpha:.2f}"
            self.ax.set_title(title)
            
        # Draw Legend for Trajectory Plot
        # Ensure we only have unique labels
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Draw Error Subplot
        if hasattr(self, "time_steps") and len(self.err_history) > 0:
            times = np.arange(len(self.err_history)) * self.dt
            self.ax_err.plot(times, self.err_history, 'b-', label='Tracking Error')
            if len(self.bound_history) == len(self.err_history):
                self.ax_err.plot(times, self.bound_history, 'orange', linestyle='--', label='Theoretical Bound')
            if len(self.rt_bound_history) == len(self.err_history):
                self.ax_err.plot(times, self.rt_bound_history, 'red', linestyle=':', label='Real-time Bound')
                
            self.ax_err.set_xlabel("Time (s)")
            self.ax_err.set_ylabel("Error Norm")
            self.ax_err.set_title("Tracking Error vs Bounds")
            self.ax_err.grid(True, alpha=0.3)
            self.ax_err.set_yscale('log')
            self.ax_err.legend(loc='upper right')
        
        if mode == "human":
            import matplotlib.pyplot as plt
            plt.draw()
            plt.pause(0.001)
        elif mode == "rgb_array":
            import io
            from PIL import Image
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            return img[:, :, :3].copy()  # Return a copy of RGB so it doesn't share memory with the canvas buffer

    def get_rewards(self, u):
        error = self.x_t - self.xref[self.time_steps]
        error = self.wrap_angles(error)

        policy_name = self.policy.__class__.__name__.lower() if hasattr(self, "policy") else ""
        if policy_name == "carl" and self.reward_mode != "inverse":
            import torch
            with torch.no_grad():
                x_tensor = torch.tensor(self.x_t, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                W = self.policy.CMG(x_tensor)[0].squeeze(0).cpu().numpy()
                M = W.T @ W
                tracking_error = error.T @ M @ error
                
            # Running average normalization
            if not hasattr(self, "carl_err_mean"):
                self.carl_err_mean = 0.0
                self.carl_err_var = 1.0
                self.carl_err_count = 1e-4

            delta = tracking_error - self.carl_err_mean
            self.carl_err_mean += delta / self.carl_err_count
            self.carl_err_var += delta * (tracking_error - self.carl_err_mean)
            self.carl_err_count += 1
            
            var = self.carl_err_var / (self.carl_err_count - 1) if self.carl_err_count > 1 else 1.0
            tracking_error = tracking_error / (np.sqrt(var) + 1e-8)
        else:
            tracking_error = (
                np.linalg.norm(
                    error,
                    ord=2,
                )
                ** 2
            )
        control_effort = np.linalg.norm(u, ord=2) ** 2

        tracking_reward = -self.tracking_scaler * tracking_error
        control_reward = -self.control_scaler * control_effort

        if self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + abs(tracking_reward))
            control_reward = 1 / (1 + abs(control_reward))

        reward = (0.5 * tracking_reward) + (0.5 * control_reward)

        return reward, {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }

    def get_rollout(self, buffer_size: int, mode: str):
        """
        Mode: Specifies whether the rollout is for training or evaluation.
            - Offline: fully offline case where we use reference control to generate data.
        """
        if mode == "c3m":
            c3m_data = dict(
                x=np.full((buffer_size, self.num_dim_x), np.nan, dtype=np.float32),
                xref=np.full((buffer_size, self.num_dim_x), np.nan, dtype=np.float32),
                uref=np.full(
                    (buffer_size, self.num_dim_control), np.nan, dtype=np.float32
                ),
            )

            # Sample all references at once
            xref = (self.X_MAX - self.X_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.X_MIN.flatten()
            uref = (self.UREF_MAX - self.UREF_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_control
            ) + self.UREF_MIN.flatten()
            xe = (self.XE_MAX - self.XE_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.XE_MIN.flatten()

            # Compose states
            x = xe + xref
            x = np.clip(x, self.X_MIN.flatten(), self.X_MAX.flatten())

            # Store
            c3m_data["x"] = x.astype(np.float32)
            c3m_data["xref"] = xref.astype(np.float32)
            c3m_data["uref"] = uref.astype(np.float32)

            # Check for NaNs
            if np.any(np.isnan(c3m_data["x"])):
                print("NaN values found in x")

            return c3m_data

        else:
            dynamics_data = dict(
                x=np.full(
                    ((buffer_size + self.max_episode_len, self.num_dim_x)),
                    np.nan,
                    dtype=np.float32,
                ),
                u=np.full(
                    ((buffer_size + self.max_episode_len, self.num_dim_control)),
                    np.nan,
                    dtype=np.float32,
                ),
                x_dot=np.full(
                    (buffer_size + self.max_episode_len, self.num_dim_x),
                    np.nan,
                    dtype=np.float32,
                ),
            )

            # === DATA FOR DYNAMICS LEARNING === #
            n_control_per_x = 3
            batch_size = ceil(buffer_size / n_control_per_x)

            if self.sample_mode == "Gaussian":
                # Compute mean and std for Gaussian distribution
                x_mean = (self.X_MAX.flatten() + self.X_MIN.flatten()) / 2.0
                x_std = (
                    self.X_MAX.flatten() - self.X_MIN.flatten()
                ) / 6.0  # 3σ covers range

                u_mean = (self.UREF_MAX.flatten() + self.UREF_MIN.flatten()) / 2.0
                u_std = (self.UREF_MAX.flatten() - self.UREF_MIN.flatten()) / 6.0

                # Sample Gaussian-distributed data
                x = np.random.normal(
                    loc=x_mean,
                    scale=x_std,
                    size=(batch_size, len(x_mean)),
                )

                u = np.random.normal(
                    loc=u_mean,
                    scale=u_std,
                    size=(batch_size, len(u_mean)),
                )

                # Step 1: Repeat x n_control_per_x times along axis 0
                x = np.concatenate([x] * n_control_per_x, axis=0)

                # Step 2: Shuffle u independently n_control_per_x times and stack
                u = np.concatenate(
                    [u[np.random.permutation(len(u))] for _ in range(n_control_per_x)],
                    axis=0,
                )

                x_dot = self.get_dynamics(x, u)

                dynamics_data["x"][:buffer_size] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"][:buffer_size] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"][:buffer_size] = x_dot[:buffer_size].astype(
                    np.float32
                )

            elif self.sample_mode == "Uniform":
                # Original sampling
                x = np.random.uniform(
                    low=self.X_MIN.flatten(),
                    high=self.X_MAX.flatten(),
                    size=(batch_size, len(self.X_MAX.flatten())),
                )
                u = np.random.uniform(
                    low=self.UREF_MIN.flatten(),
                    high=self.UREF_MAX.flatten(),
                    size=(batch_size, len(self.UREF_MAX.flatten())),
                )

                # Step 1: Repeat x n_control_per_x times along axis 0
                x = np.concatenate([x] * n_control_per_x, axis=0)

                # Step 2: Shuffle u independently n_control_per_x times and stack
                u = np.concatenate(
                    [u[np.random.permutation(len(u))] for _ in range(n_control_per_x)],
                    axis=0,
                )

                x_dot = self.get_dynamics(x, u)

                dynamics_data["x"][:buffer_size] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"][:buffer_size] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"][:buffer_size] = x_dot[:buffer_size].astype(
                    np.float32
                )
            else:
                current_time = 0
                while current_time < buffer_size:
                    xref_0, _, x_0 = self.define_initial_state()

                    freqs = list(range(1, 11))
                    weights = np.random.randn(len(freqs), len(self.UREF_MIN))
                    weights = (
                        weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
                    ).tolist()

                    x_t, x_t_wrapped = x_0.copy(), x_0.copy()
                    for i, _t in enumerate(self.t):
                        u_t = self.sample_reference_controls(
                            freqs, weights, _t, {"xref_0": xref_0}, add_noise=True
                        )
                        next_x, next_x_wrapped, term, _, x_dot = self.get_transition(
                            x_t, u_t
                        )

                        # clip state to bounds
                        next_x_wrapped = np.clip(
                            next_x_wrapped, self.X_MIN.flatten(), self.X_MAX.flatten()
                        )

                        ### LOGGING ###
                        dynamics_data["x"][current_time + i] = x_t_wrapped
                        dynamics_data["u"][current_time + i] = u_t
                        dynamics_data["x_dot"][current_time + i] = x_dot

                        x_t = next_x
                        x_t_wrapped = next_x_wrapped

                        # here trunc is not necessary since we use for loops.
                        if term:
                            break

                    current_time += i + 1

                dynamics_data["x"] = dynamics_data["x"][:buffer_size]
                dynamics_data["u"] = dynamics_data["u"][:buffer_size]
                dynamics_data["x_dot"] = dynamics_data["x_dot"][:buffer_size]

            return dynamics_data
