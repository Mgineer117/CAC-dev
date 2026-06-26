import os

import numpy as np
import torch
import torch.nn as nn

from envs.env_base import BaseEnv

## STATE
# x = [p_x, p_y, phi, theta, psi, q_18]
## CONTROL
# u = [q_18_dot]

# Denote angle indices to handle smooth transition
ANGLE_IDX = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
]


# Jethexa PARAMETERS
"""
--- STATE BOUNDS ---
Base Pos X     : Min =  -4.7857           , Max =  -0.3236
Base Pos Y     : Min =  -1.9181           , Max =   2.1212
Base Pos Z     : Min =   0.1659           , Max =   0.1883
Base Roll      : Min =  -0.0286 ( -pi/110), Max =   0.0157 (  pi/200)
Base Pitch     : Min =  -0.0365 (  -pi/86), Max =   0.0431 (   pi/73)
Base Yaw       : Min =  -3.1415 (     -pi), Max =   3.1412 (      pi)
State Joint 1  : Min =  -0.0744 (  -pi/42), Max =   0.5101 (    pi/6)
State Joint 2  : Min =   0.6465 (    pi/5), Max =   1.0684 (    pi/3)
State Joint 3  : Min =  -0.8114 (   -pi/4), Max =  -0.3057 (  -pi/10)
State Joint 4  : Min =  -0.3379 (   -pi/9), Max =   0.3325 (    pi/9)
State Joint 5  : Min =   0.7230 (    pi/4), Max =   1.0689 (    pi/3)
State Joint 6  : Min =  -0.7576 (   -pi/4), Max =  -0.5199 (   -pi/6)
State Joint 7  : Min =  -0.5044 (   -pi/6), Max =   0.0679 (   pi/46)
State Joint 8  : Min =   0.6466 (    pi/5), Max =   1.0671 (    pi/3)
State Joint 9  : Min =  -0.8039 (   -pi/4), Max =  -0.3062 (  -pi/10)
State Joint 10 : Min =  -0.0678 (  -pi/46), Max =   0.5042 (    pi/6)
State Joint 11 : Min =   0.6466 (    pi/5), Max =   1.0671 (    pi/3)
State Joint 12 : Min =  -0.8039 (   -pi/4), Max =  -0.3062 (  -pi/10)
State Joint 13 : Min =  -0.3324 (   -pi/9), Max =   0.3378 (    pi/9)
State Joint 14 : Min =   0.7230 (    pi/4), Max =   1.0689 (    pi/3)
State Joint 15 : Min =  -0.7576 (   -pi/4), Max =  -0.5201 (   -pi/6)
State Joint 16 : Min =  -0.5100 (   -pi/6), Max =   0.0742 (   pi/42)
State Joint 17 : Min =   0.6464 (    pi/5), Max =   1.0684 (    pi/3)
State Joint 18 : Min =  -0.8115 (   -pi/4), Max =  -0.3057 (  -pi/10)

--- CONTROL BOUNDS ---
Control Joint 1  : Min =  -0.2169, Max =   0.1777
Control Joint 2  : Min =  -0.2767, Max =   0.2621
Control Joint 3  : Min =  -0.1425, Max =   0.3161
Control Joint 4  : Min =  -0.2147, Max =   0.3097
Control Joint 5  : Min =  -0.2364, Max =   0.3030
Control Joint 6  : Min =  -0.1669, Max =   0.1456
Control Joint 7  : Min =  -0.3043, Max =   0.1898
Control Joint 8  : Min =  -0.2581, Max =   0.2709
Control Joint 9  : Min =  -0.2187, Max =   0.1806
Control Joint 10 : Min =  -0.2074, Max =   0.1441
Control Joint 11 : Min =  -0.2301, Max =   0.3411
Control Joint 12 : Min =  -0.2863, Max =   0.2615
Control Joint 13 : Min =  -0.1681, Max =   0.3005
Control Joint 14 : Min =  -0.2627, Max =   0.2637
Control Joint 15 : Min =  -0.1517, Max =   0.1463
Control Joint 16 : Min =  -0.2927, Max =   0.1621
Control Joint 17 : Min =  -0.2451, Max =   0.2528
Control Joint 18 : Min =  -0.1961, Max =   0.2144
"""

# X bounds
X_MIN = np.array(
    [
        -5.0,
        -2.5,
        -np.pi / 110,  # phi
        -np.pi / 86,  # theta
        -np.pi / 1,  # psi
        -np.pi / 42,  # q1
        np.pi / 5,  # q2
        -np.pi / 4,  # q3
        -np.pi / 9,  # q4
        np.pi / 4,  # q5
        -np.pi / 4,  # q6
        -np.pi / 6,  # q7
        np.pi / 5,  # q8
        -np.pi / 4,  # q9
        -np.pi / 46,  # q10
        np.pi / 5,  # q11
        -np.pi / 4,  # q12
        -np.pi / 9,  # q13
        np.pi / 4,  # q14
        -np.pi / 4,  # q15
        -np.pi / 6,  # q16
        np.pi / 5,  # q17
        -np.pi / 4,  # q18
    ]
).reshape(-1, 1)

X_MAX = np.array(
    [
        0.0,
        2.5,
        np.pi / 200,  # phi
        np.pi / 73,  # theta
        np.pi / 1,  # psi
        np.pi / 6,  # q1
        np.pi / 3,  # q2
        -np.pi / 10,  # q3
        np.pi / 9,  # q4
        np.pi / 3,  # q5
        -np.pi / 6,  # q6
        np.pi / 46,  # q7
        np.pi / 3,  # q8
        -np.pi / 10,  # q9
        np.pi / 6,  # q10
        np.pi / 3,  # q11
        -np.pi / 10,  # q12
        np.pi / 9,  # q13
        np.pi / 3,  # q14
        -np.pi / 6,  # q15
        np.pi / 42,  # q16
        np.pi / 3,  # q17
        -np.pi / 10,  # q18
    ]
).reshape(-1, 1)

# Initial reference state perturbation bounds (XE_INIT)
# Offsets dictate how "far away" from the reference trajectory the robot spawns.

base_pos_offset = 0.15  # +/- 15 cm error in starting X/Y position
roll_pitch_offset = np.pi / 400  # +/- 0.45 degrees (Must be tiny! Bounds are ~ pi/80)
yaw_offset = np.pi / 12  # +/- 15 degrees of heading error
joint_offset = np.pi / 60  # +/- 3 degrees of joint position error

XE_INIT_MAX = np.array(
    [
        base_pos_offset,  # 0: Base Pos X
        base_pos_offset,  # 1: Base Pos Y
        roll_pitch_offset,  # 2: Base Roll (phi)
        roll_pitch_offset,  # 3: Base Pitch (theta)
        yaw_offset,  # 4: Base Yaw (psi)
    ]
    + [joint_offset] * 18  # 5-22: 18 Leg Joints
).reshape(-1, 1)

XE_INIT_MIN = -XE_INIT_MAX  # Symmetrical negative bounds

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = XE_INIT_MIN
XE_MAX = XE_INIT_MAX

# reference control bounds
UREF_MIN = np.array(
    [
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
        -0.3,
    ]
).reshape(-1, 1)
UREF_MAX = np.array(
    [
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
    ]
).reshape(-1, 1)

env_config = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    # "xref_init_min": XREF_INIT_MIN,
    # "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "angle_idx": ANGLE_IDX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 23,
    "num_dim_control": 18,
    "pos_dimension": 2,
    "dt": 0.1,
    "time_bound": 30.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.0,  # control cost weight
}


class JethexaEnv(BaseEnv):
    def __init__(
        self,
        sample_mode: str = "uniform",
        reward_mode: str = "default",
    ) -> None:
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "jethexa"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["reward_mode"] = reward_mode

        super().__init__(env_config)

        self.base_dim = 5

        self.jacobian_net = nn.Sequential(
            nn.Linear(self.num_dim_x, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.base_dim * self.num_dim_control),
        )

        # Load the weights
        self.jacobian_net.load_state_dict(
            torch.load("model/Jethexa_jacobian_net.pth", map_location="cpu")
        )

        # 1. Freeze the weights (Disables gradient tracking forever)
        for param in self.jacobian_net.parameters():
            param.requires_grad = False

        # 2. Set to evaluation mode (Best practice for inference)
        self.jacobian_net.eval()

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        f = lib.zeros((n, self.num_dim_x))
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]

        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        if lib == torch:
            B[:, :5, :] = self.jacobian_net(x)
            B[:, 5:, :] = torch.eye(self.num_dim_control).unsqueeze(0).repeat(n, 1, 1)

        else:
            B[:, :5, :] = self.jacobian_net(torch.from_numpy(x)).cpu().numpy()
            B[:, 5:, :] = np.eye(self.num_dim_control).unsqueeze(0).repeat(n, 1, 1)

        return B

    def _B_null_logic(self, x, n, lib):
        """
        Calculates the orthogonal complement B_null(x) (or B_bot).
        Ensures that transpose(B) * B_bot = 0.
        """
        if lib == torch:
            # 1. Get J(x) from the network: shape (n, 5, m)
            J = self.jacobian_net(x)

            # 2. Transpose J to match the bottom block: shape (n, m, 5)
            J_T = J.transpose(1, 2)

            # 3. Create the 5x5 Identity matrix batch: shape (n, 5, 5)
            I = torch.eye(5, device=J.device).unsqueeze(0).repeat(n, 1, 1)

            # 4. Concatenate [I; -J_T] along the row dimension (dim=1)
            # Resulting shape: (n, 5 + m, 5) -> (n, num_dim_x, num_dim_x - num_dim_control)
            Bbot = torch.cat((I, -J_T), dim=1)

        else:  # lib == np
            # 1. Get J(x) from the network and convert to numpy
            J = self.jacobian_net(torch.from_numpy(x)).cpu().numpy()  # shape (n, 5, m)

            # 2. Transpose J: shape (n, m, 5)
            J_T = np.transpose(J, (0, 2, 1))

            # 3. Create the 5x5 Identity matrix batch: shape (n, 5, 5)
            I = np.eye(5)
            I_batch = np.repeat(I[np.newaxis, :, :], n, axis=0)

            # 4. Concatenate [I; -J_T] along the row dimension (axis=1)
            Bbot = np.concatenate((I_batch, -J_T), axis=1)

        return Bbot

    def reset(self, seed=None, options: dict | None = None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.time_steps = 0

        # Custom reset behavior that keeps reference trajectory but changes initial state
        if (
            (options or {}).get("replace_x_0", False)
            and hasattr(self, "xref")
            and hasattr(self, "uref")
        ):
            xe_0 = np.random.uniform(
                self.xe_init_min.flatten(), self.xe_init_max.flatten()
            )
            self.x_t = self.xref[0] + xe_0
        else:
            # Default reset behavior
            self.x_t, self.xref, self.uref, self.episode_len = self.system_reset(
                options
            )

        state = self.construct_state()
        self.init_tracking_error = np.linalg.norm(self.x_t - self.xref[0], ord=2) ** 2

        return state, {"x": self.x_t, "tracking_error": self.init_tracking_error}

    def system_reset(self, options=None):
        """Resets the system to an initial state and generates a reference trajectory."""
        # xref_0, xe_0, x_0 = self.define_initial_state()

        if options and options.get("eval_mode", False):
            traj_path = "jethexa_data/validation/"
        else:
            traj_path = "jethexa_data/training/"

        # sample one from traj_path
        traj_files = [f for f in os.listdir(traj_path) if f.endswith(".npz")]
        traj_file = np.random.choice(traj_files)
        traj_data = np.load(os.path.join(traj_path, traj_file))

        xref = traj_data["states"][-15:]
        uref = traj_data["controls"][-15:]

        assert (
            xref.shape[0] == uref.shape[0]
        ), "Trajectory length mismatch between states and controls."
        assert (
            xref.shape[0] == 300 and uref.shape[0] == 300
        ), "Expected trajectory length of 300 time steps."

        x_0 = xref[0] + np.random.uniform(
            self.XE_INIT_MIN.flatten(), self.XE_INIT_MAX.flatten()
        )

        return (
            x_0,
            xref,
            uref,
            300,
        )
