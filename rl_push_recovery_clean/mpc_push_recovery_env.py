import numpy as np
import mujoco as mj
import gymnasium as gym
from gymnasium import spaces

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.inplace_terrain_v3 import get_flat_model

GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

MPC_DT = GAIT_T / 16
STEPS_PER_MPC = 2

TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}

V_X_NOM = 0.6
NOMINAL_HEIGHT = 0.27
MAX_VY_CORRECTION = 0.4
MAX_ROLL_CORRECTION_DEG = 10.0
ALPHA = 0.3

BASE_BODY_ID = 1
PUSH_TIME = 2.0
PUSH_DURATION_S = 0.1
RECOVERY_WINDOW_S = 1.25

# Event-trigger gating thresholds, per external review: 1.5x the
# measured nominal (0N) walking maximum, so normal gait never triggers
WX_TRIGGER_DEG_S = 56.85
VY_TRIGGER = 0.188
ROLL_TRIGGER_DEG = 2.08

EPISODE_LENGTH_S = 6.0

PUSH_BUCKETS = [
    (80, 120, 0.30),
    (120, 160, 0.50),
    (160, 180, 0.20),
]

W_ROLL = 2.0
W_ROLL_RATE = 0.1
W_VY = 1.0
W_ACTION = 0.05
W_ACTION_RATE = 0.5
FALL_PENALTY = -500.0
INFEASIBLE_PENALTY = -20.0
RECOVERY_BONUS = 100.0
ROLL_TOL_DEG = 3.0
ROLL_RATE_TOL_DEG_S = 10.0
VY_TOL = 0.05


class MPCPushRecoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._last_correction = np.zeros(2, dtype=np.float32)
        self._applied_correction = np.zeros(2)
        self._episode_reward = 0.0

    def _sample_push(self):
        r = self.np_random.uniform(0, 1)
        cum = 0.0
        for lo, hi, p in PUSH_BUCKETS:
            cum += p
            if r < cum:
                mag = self.np_random.uniform(lo, hi)
                break
        else:
            mag = self.np_random.uniform(*PUSH_BUCKETS[-1][:2])
        sign = self.np_random.choice([-1.0, 1.0])
        return sign * mag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._push_force = self._sample_push()
        self._push_applied = False

        model = get_flat_model()
        self.go2 = PinGo2Model()
        self.mujoco_go2 = MuJoCo_GO2_Model(model=model)
        self.leg_ctrl = LegController()
        self.traj = ComTraj(self.go2)
        self.gait = Gait(GAIT_HZ, GAIT_DUTY)

        q_init = self.go2.current_config.get_q()
        q_init[0], q_init[1] = 0.0, 0.0
        self.mujoco_go2.update_with_q_pin(q_init)
        self.mujoco_go2.model.opt.timestep = SIM_DT

        self.traj.generate_traj(self.go2, self.gait, 0.0, V_X_NOM, 0.0, NOMINAL_HEIGHT, 0.0,
                                 time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=0.0)
        self.mpc = CentroidalMPC(self.go2, self.traj)
        self.U_opt = np.zeros((12, self.traj.N), dtype=float)

        self._last_correction = np.zeros(2, dtype=np.float32)
        self._applied_correction = np.zeros(2)
        self._episode_reward = 0.0
        self._ctrl_i = 0
        self._n_mpc_infeasible = 0
        self._n_wbc_infeasible = 0
        self._n_decision_steps = 0
        self._recovery_evaluated = False
        self._gate_open = False
        self._gate_open_until = -1.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        self.mujoco_go2.update_pin_with_mujoco(self.go2)
        d = self.mujoco_go2.data

        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))

        R = self.go2.R_world_to_body
        g_world = np.array([0, 0, -1.0])
        g_body = R @ g_world

        v_body = self.go2.current_config.base_vel
        w_body = self.go2.current_config.base_ang_vel

        current_mask = self.gait.compute_current_mask(float(d.time)).reshape(4,).astype(np.float32)

        obs = np.concatenate([
            g_body, [roll, pitch], v_body, w_body, current_mask, self._last_correction
        ]).astype(np.float32)
        return obs

    def _update_gate(self, roll_deg, wx_deg_s, vy):
        """Event-triggered gate: opens when disturbance detected
        (|wx|, |vy|, or |roll| exceeds nominal-walking thresholds),
        stays open for RECOVERY_WINDOW_S, per external review."""
        t_now = float(self.mujoco_go2.data.time)
        if not self._gate_open and (
            abs(wx_deg_s) > WX_TRIGGER_DEG_S or
            abs(vy) > VY_TRIGGER or
            abs(roll_deg) > ROLL_TRIGGER_DEG
        ):
            self._gate_open = True
            self._gate_open_until = t_now + RECOVERY_WINDOW_S
        if self._gate_open and t_now > self._gate_open_until:
            self._gate_open = False
        return self._gate_open

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).copy()

        d0 = self.mujoco_go2.data
        qw0, qx0, qy0, qz0 = d0.qpos[3:7]
        roll_now = np.degrees(np.arctan2(2*(qw0*qx0+qy0*qz0), 1-2*(qx0**2+qy0**2)))
        wx_now = np.degrees(self.go2.current_config.base_ang_vel[0])
        vy_now = self.go2.current_config.base_vel[1]
        gate_open = self._update_gate(roll_now, wx_now, vy_now)

        if gate_open:
            raw_vy_corr = float(action[0]) * MAX_VY_CORRECTION
            raw_roll_corr = float(action[1]) * MAX_ROLL_CORRECTION_DEG
        else:
            raw_vy_corr = 0.0
            raw_roll_corr = 0.0

        raw = np.array([raw_vy_corr, raw_roll_corr])
        applied = (1 - ALPHA) * self._applied_correction + ALPHA * raw
        self._applied_correction = applied
        self._last_correction = np.array([
            applied[0] / MAX_VY_CORRECTION,
            applied[1] / MAX_ROLL_CORRECTION_DEG,
        ], dtype=np.float32)
        self._n_decision_steps += 1

        v_y_cmd = applied[0]
        roll_ref_deg = applied[1]

        mpc_solve_failed = False
        wbc_solve_failed = False

        for _ in range(STEPS_PER_MPC):
            time_now_s = float(self.mujoco_go2.data.time)

            if PUSH_TIME <= time_now_s < PUSH_TIME + PUSH_DURATION_S:
                self.mujoco_go2.data.xfrc_applied[BASE_BODY_ID, 1] = self._push_force
                self._push_applied = True
            else:
                self.mujoco_go2.data.xfrc_applied[BASE_BODY_ID, :] = 0.0

            if self._ctrl_i % STEPS_PER_MPC == 0:
                self.traj.generate_traj(self.go2, self.gait, time_now_s, V_X_NOM, v_y_cmd, NOMINAL_HEIGHT,
                                         0.0, time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=roll_ref_deg)
                sol = self.mpc.solve_QP(self.go2, self.traj, False)
                if not self.mpc.last_solve_success:
                    mpc_solve_failed = True
                    self._n_mpc_infeasible += 1
                N = self.traj.N
                w_opt = sol["x"].full().flatten()
                self.U_opt = w_opt[12*N:].reshape((12, N), order="F")

            mpc_force = self.U_opt[:, 0]
            leg_outputs = self.leg_ctrl.compute_all_torques(
                self.go2, self.gait, mpc_force, time_now_s, use_wbc=True, slope_deg=0.0)
            if not getattr(self.leg_ctrl, "last_wbc_success", True):
                wbc_solve_failed = True
                self._n_wbc_infeasible += 1

            tau_raw = np.zeros(12)
            for leg in LEG_NAMES:
                tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
            tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)

            for _ in range(CTRL_DECIM):
                mj.mj_step1(self.mujoco_go2.model, self.mujoco_go2.data)
                self.mujoco_go2.set_joint_torque(tau_hold)
                mj.mj_step2(self.mujoco_go2.model, self.mujoco_go2.data)

            self._ctrl_i += 1
            self.mujoco_go2.update_pin_with_mujoco(self.go2)

        d = self.mujoco_go2.data
        base_z = d.qpos[2]
        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
        wx = self.go2.current_config.base_ang_vel[0]
        v_y_actual = self.go2.current_config.base_vel[1]

        terminated = False
        term_reason = None
        if mpc_solve_failed:
            terminated = True
            term_reason = "mpc_infeasible"
        elif base_z < 0.15:
            terminated = True
            term_reason = "low_height"
        elif abs(roll) > 0.8:
            terminated = True
            term_reason = "extreme_roll"
        elif abs(pitch) > 0.8:
            terminated = True
            term_reason = "extreme_pitch"

        truncated = self._ctrl_i * CTRL_DT >= EPISODE_LENGTH_S
        if truncated and not terminated:
            term_reason = "success_timeout"

        reward = 0.0
        reward -= W_ROLL * roll ** 2
        reward -= W_ROLL_RATE * (wx) ** 2
        reward -= W_VY * v_y_actual ** 2
        reward -= W_ACTION * np.sum((applied / np.array([MAX_VY_CORRECTION, MAX_ROLL_CORRECTION_DEG])) ** 2)
        reward -= W_ACTION_RATE * np.sum((raw - self._applied_correction) ** 2)

        t_now2 = float(self.mujoco_go2.data.time)
        if self._push_applied and not self._recovery_evaluated and self._gate_open_until > 0 and t_now2 >= self._gate_open_until:
            self._recovery_evaluated = True
            if (abs(np.degrees(roll)) < ROLL_TOL_DEG and
                abs(np.degrees(wx)) < ROLL_RATE_TOL_DEG_S and
                abs(v_y_actual) < VY_TOL):
                reward += RECOVERY_BONUS

        if terminated and term_reason in ("low_height", "extreme_roll", "extreme_pitch"):
            reward += FALL_PENALTY
        if mpc_solve_failed or wbc_solve_failed:
            reward += INFEASIBLE_PENALTY
        self._episode_reward += reward

        info = {}
        if terminated or truncated:
            n = max(self._n_decision_steps, 1)
            info["residual_episode"] = {
                "l": self._ctrl_i,
                "r": self._episode_reward,
                "term_reason": term_reason,
                "push_force_n": self._push_force,
                "frac_mpc_infeasible": self._n_mpc_infeasible / n,
                "frac_wbc_infeasible": self._n_wbc_infeasible / n,
                "final_roll_deg": np.degrees(roll),
                "final_wx_deg_s": wx,
                "final_vy": v_y_actual,
            }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
