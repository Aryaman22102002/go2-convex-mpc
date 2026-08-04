"""
mpc_speed_residual_env.py

Speed-residual environment, per external review's design following the
forward-speed sweep's clean, monotonic confirmation that commanded speed
is the dominant control lever for genuine-uphill survival.

Action: [Delta_v_x] in [-1, 1] raw, mapped to a BOUNDED, NEGATIVE-ONLY
speed reduction Delta_v_x in [-MAX_SPEED_REDUCTION, 0] -- per review,
deliberately not allowing speed INCREASES initially, since the sweep only
validated the "slower is safer" direction; allowing increases would let
the policy explore unvalidated, likely-unhelpful territory.

v_x_cmd = V_X_NOM + Delta_v_x, rate-limited via the same low-pass filter
pattern used for the pitch residual (abrupt speed-reference changes can
destabilize MPC even when the target value itself is sensible).

Reward balances forward progress against unnecessary slowdown -- a
survival-only reward would trivially learn v_x_cmd = V_X_NOM -
MAX_SPEED_REDUCTION everywhere, per review's warning.
"""
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
from convex_mpc.inplace_terrain import get_universal_model, set_flat, set_slope, set_step

GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}

V_X_NOM = 0.6
MAX_SPEED_REDUCTION = 0.4   # v_x_cmd in [0.2, 0.6], matching the validated sweep range
ALPHA = 0.3                  # low-pass filter on applied correction, matching pitch residual

EPISODE_LENGTH_S = 6.0

# Reward weights -- initial, reasonable defaults; NOT yet empirically
# tuned. Per review: must reward forward progress, not just survival, or
# the policy trivially learns max-slowdown everywhere.
W_PROGRESS = 10.0    # reward per unit of ACTUAL forward velocity achieved
W_TRACKING_ERR = 2.0  # penalize v_x_actual deviating from v_x_cmd (tracking quality)
W_RESIDUAL = 0.3      # penalize |Delta_v_x| magnitude (only slow down when needed) --
                      # reduced from 1.0: at the old weight, slowing down cost reward
                      # EVERY step, on top of the direct W_PROGRESS cost of lower v_x,
                      # a double penalty that left no incentive to ever use this channel
W_RATE = 2.0          # penalize step-to-step change in applied correction
FALL_PENALTY = -500.0  # was -50: a failing-but-fast episode (e.g. 59 steps at
                       # ~6/step =~354, -50 =~304) was NOT meaningfully worse than
                       # succeeding slowly -- raised so failure genuinely dominates
                       # the accumulated progress-reward "savings" from never slowing down
SUCCESS_BONUS = 300.0  # was implicit/zero: makes "slow but complete" unambiguously
                       # better than "fast but fails partway", rather than relying
                       # on the fall penalty alone to make that comparison work out
INFEASIBLE_PENALTY = -20.0


class MPCSpeedResidualEnv(gym.Env):
    """Residual correction on top of the existing MPC+WBC pipeline, on
    COMMANDED FORWARD SPEED rather than pitch reference. Action:
    [speed_correction] in [-1, 1], mapped to Delta_v_x in
    [-MAX_SPEED_REDUCTION, 0] (bounded, negative-only per review).
    At action=-1 (mapped to Delta_v_x=0), reproduces the nominal
    fixed-V_X_NOM MPC+WBC controller exactly.
    """

    metadata = {"render_modes": []}

    def __init__(self, terrain_curriculum=None, uphill_frac=0.8):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.terrain_curriculum = terrain_curriculum or {"flat": 0.34, "slope": 0.33, "step": 0.33}
        self.uphill_frac = uphill_frac

        self._last_correction = np.zeros(1, dtype=np.float32)
        self._applied_correction = np.zeros(1)
        self._episode_reward = 0.0
        self._n_decision_steps = 0
        self._n_raw_speed_saturated = 0
        self._n_mpc_infeasible = 0
        self._n_wbc_infeasible = 0

    def _sample_terrain(self):
        names = list(self.terrain_curriculum.keys())
        probs = np.array([self.terrain_curriculum[n] for n in names])
        probs = probs / probs.sum()
        choice = self.np_random.choice(names, p=probs)

        model = get_universal_model()
        if choice == "flat":
            set_flat(model)
            return "flat", 0.0
        elif choice == "slope":
            if self.np_random.uniform(0, 1) < self.uphill_frac:
                slope_deg = self.np_random.uniform(-15, -2)   # genuine uphill
            else:
                slope_deg = self.np_random.uniform(2, 15)     # genuine downhill (regression check)
            set_slope(model, slope_deg)
            return "slope", slope_deg
        else:
            direction = self.np_random.choice(["step_up", "step_down"])
            step_height = self.np_random.uniform(0.03, 0.08)
            set_step(model, step_height, direction="up" if direction == "step_up" else "down")
            return direction, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        terrain_kind, true_slope_deg = self._sample_terrain()
        self._terrain_kind = terrain_kind
        self._true_slope_deg = true_slope_deg

        self.go2 = PinGo2Model()
        self.mujoco_go2 = MuJoCo_GO2_Model(model=get_universal_model())
        self.leg_ctrl = LegController()
        self.traj = ComTraj(self.go2)
        self.gait = Gait(GAIT_HZ, GAIT_DUTY)

        q_init = self.go2.current_config.get_q()
        q_init[0], q_init[1] = 0.0, 0.0
        self.mujoco_go2.update_with_q_pin(q_init)
        self.mujoco_go2.model.opt.timestep = SIM_DT

        self.traj.generate_traj(self.go2, self.gait, 0.0, V_X_NOM, 0.0, 0.27, 0.0,
                                 time_step=MPC_DT, slope_deg=0.0)
        self.mpc = CentroidalMPC(self.go2, self.traj)
        self.U_opt = np.zeros((12, self.traj.N), dtype=float)

        self._last_correction = np.zeros(1, dtype=np.float32)
        self._applied_correction = np.zeros(1)
        self._episode_reward = 0.0
        self._n_decision_steps = 0
        self._n_raw_speed_saturated = 0
        self._n_mpc_infeasible = 0
        self._n_wbc_infeasible = 0
        self._ctrl_i = 0

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

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).copy()
        # Map [-1, 1] -> [-MAX_SPEED_REDUCTION, 0]: action=-1 -> 0 (no
        # reduction, reproduces nominal), action=+1 -> -MAX_SPEED_REDUCTION
        raw_speed_corr = -MAX_SPEED_REDUCTION * (float(action[0]) + 1.0) / 2.0

        applied_speed_corr = (1 - ALPHA) * self._applied_correction[0] + ALPHA * raw_speed_corr
        self._applied_correction = np.array([applied_speed_corr])
        self._last_correction = np.array(
            [(applied_speed_corr + MAX_SPEED_REDUCTION / 2.0) / (MAX_SPEED_REDUCTION / 2.0)],
            dtype=np.float32)

        self._n_decision_steps += 1
        if action[0] > 0.95:
            self._n_raw_speed_saturated += 1

        v_x_cmd = V_X_NOM + applied_speed_corr

        mpc_solve_failed = False
        wbc_solve_failed = False

        for _ in range(STEPS_PER_MPC):
            time_now_s = float(self.mujoco_go2.data.time)

            if self._ctrl_i % STEPS_PER_MPC == 0:
                self.traj.generate_traj(self.go2, self.gait, time_now_s, v_x_cmd, 0.0, 0.27, 0.0,
                                         time_step=MPC_DT, slope_deg=0.0)
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
        base_x, base_z = d.qpos[0], d.qpos[2]
        slope_rad = np.radians(self._true_slope_deg)
        terrain_h = -base_x * np.tan(slope_rad) if self._terrain_kind == "slope" else 0.0
        height_rel = base_z - terrain_h
        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
        pitch_rel = pitch - slope_rad if self._terrain_kind == "slope" else pitch

        v_x_actual = self.go2.current_config.base_vel[0]

        terminated = False
        term_reason = None
        if mpc_solve_failed:
            terminated = True
            term_reason = "mpc_infeasible"
        elif height_rel < 0.15:
            terminated = True
            term_reason = "low_height"
        elif abs(roll) > 0.8:
            terminated = True
            term_reason = "extreme_roll"
        elif abs(pitch_rel) > 0.8:
            terminated = True
            term_reason = "extreme_pitch"

        truncated = self._ctrl_i * CTRL_DT >= EPISODE_LENGTH_S
        if truncated and not terminated:
            term_reason = "success_timeout"

        # Reward: forward progress minus tracking error minus unnecessary
        # slowdown minus chatter -- per review, prevents the policy from
        # trivially learning max-slowdown everywhere
        reward = W_PROGRESS * v_x_actual
        reward -= W_TRACKING_ERR * (v_x_actual - v_x_cmd) ** 2
        reward -= W_RESIDUAL * applied_speed_corr ** 2
        reward -= W_RATE * (applied_speed_corr - self._applied_correction[0]) ** 2
        if terminated and term_reason in ("low_height", "extreme_roll", "extreme_pitch"):
            reward += FALL_PENALTY
        if mpc_solve_failed or wbc_solve_failed:
            reward += INFEASIBLE_PENALTY
        if truncated and term_reason == "success_timeout":
            reward += SUCCESS_BONUS
        self._episode_reward += reward

        info = {}
        if terminated or truncated:
            n = max(self._n_decision_steps, 1)
            info["residual_episode"] = {
                "l": self._ctrl_i,
                "r": self._episode_reward,
                "term_reason": term_reason,
                "terrain": self._terrain_kind,
                "true_slope_deg": self._true_slope_deg,
                "frac_speed_saturated": self._n_raw_speed_saturated / n,
                "frac_mpc_infeasible": self._n_mpc_infeasible / n,
                "frac_wbc_infeasible": self._n_wbc_infeasible / n,
                "final_v_x_cmd": v_x_cmd,
                "final_v_x_actual": v_x_actual,
            }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
