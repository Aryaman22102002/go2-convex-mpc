"""
mpc_residual_env.py

Residual RL environment: wraps the EXISTING MPC+WBC pipeline unchanged.
The RL policy does NOT replace force solving, torque computation, or any
physical constraint (friction cone, torque limits) -- all of that remains
exactly as implemented in centroidal_mpc.py / wbc_qp.py / leg_controller.py.

RL's only job: output a small, bounded correction to the reference pitch and
height fed into ComTraj.generate_traj, compensating for the diagnosed gap in
the MPC's flat-ground-assuming internal dynamics model. If the correction is
exactly zero, behavior is IDENTICAL to the unmodified MPC+WBC controller --
this is a true residual design, not a replacement.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

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

MAX_PITCH_CORRECTION_DEG = 15.0   # bounds on the residual correction itself
MAX_HEIGHT_CORRECTION_M  = 0.05

EPISODE_LENGTH_S = 6.0
X_VEL_DES = 0.6


class MPCResidualEnv(gym.Env):
    """Residual correction on top of the existing MPC+WBC pipeline.
    Action: [pitch_correction, height_correction], each in [-1, 1] before
    scaling. Reference pitch/height passed to ComTraj.generate_traj becomes
    slope_deg_estimate + pitch_correction*MAX_PITCH_CORRECTION_DEG (in the
    trajectory's own slope_deg argument) and z_pos_des_body +
    height_correction*MAX_HEIGHT_CORRECTION_M.

    Note: this does NOT give the policy the true slope_deg -- the residual
    is applied as a correction to a DEFAULT flat-ground reference (slope_deg
    input to ComTraj stays 0 unless the policy corrects it), forcing the
    policy to infer the right correction from proprioception, matching how
    the diagnosed failure (compounding drift from a flat-ground dynamics
    model) would actually need to be corrected by a controller with no
    direct terrain sensor.
    """

    metadata = {"render_modes": []}

    def __init__(self, terrain_curriculum=None, action_channels="both"):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Default curriculum: even mix of flat/slope/step, can be overridden
        # for evaluation (e.g. forcing slope-only)
        self.terrain_curriculum = terrain_curriculum or {"flat": 0.34, "slope": 0.33, "step": 0.33}

        # Ablation support (per review): "both", "pitch_only", "height_only".
        # Action space stays 2D regardless (simplest, no architecture change
        # needed); the disabled channel is forced to zero before use.
        assert action_channels in ("both", "pitch_only", "height_only")
        self.action_channels = action_channels

        self._last_correction = np.zeros(2, dtype=np.float32)
        self._applied_correction = np.zeros(2)
        self._episode_reward = 0.0

    def _sample_terrain(self):
        """Samples a terrain kind and continuous parameter, then mutates the
        shared universal model in-place -- genuinely continuous randomization
        (not a fixed discrete library), since mutation has no disk I/O cost
        and doesn't touch the from_xml_path resource-exhaustion trigger."""
        names = list(self.terrain_curriculum.keys())
        probs = np.array([self.terrain_curriculum[n] for n in names])
        probs = probs / probs.sum()
        choice = self.np_random.choice(names, p=probs)

        model = get_universal_model()
        if choice == "flat":
            set_flat(model)
            return "flat", 0.0
        elif choice == "slope":
            slope_deg = self.np_random.uniform(-15, 15)
            if abs(slope_deg) < 2.0:
                slope_deg = np.sign(slope_deg) * 2.0 if slope_deg != 0 else 5.0
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
        self._true_slope_deg = true_slope_deg  # used ONLY for fall detection, never observed by the policy

        self.go2 = PinGo2Model()
        self.mujoco_go2 = MuJoCo_GO2_Model(model=get_universal_model())
        self.leg_ctrl = LegController()
        self.traj = ComTraj(self.go2)
        self.gait = Gait(GAIT_HZ, GAIT_DUTY)

        q_init = self.go2.current_config.get_q()
        q_init[0], q_init[1] = 0.0, 0.0
        self.mujoco_go2.update_with_q_pin(q_init)
        self.mujoco_go2.model.opt.timestep = SIM_DT

        self.traj.generate_traj(self.go2, self.gait, 0.0, X_VEL_DES, 0.0, 0.27, 0.0,
                                 time_step=MPC_DT, slope_deg=0.0)
        self.mpc = CentroidalMPC(self.go2, self.traj)
        self.U_opt = np.zeros((12, self.traj.N), dtype=float)

        self._ctrl_i = 0
        self._sim_k = 0
        self._tau_hold = np.zeros(12, dtype=float)
        self._last_correction = np.zeros(2, dtype=np.float32)
        self._applied_correction = np.zeros(2)
        self._episode_reward = 0.0
        self.max_ctrl_steps = int(EPISODE_LENGTH_S * CTRL_HZ)

        # Per-episode tracking, per external review's required logging
        self._n_decision_steps = 0
        self._n_raw_pitch_saturated = 0    # |raw pitch corr| > 0.95*max
        self._n_raw_height_saturated = 0
        self._n_mpc_infeasible = 0
        self._n_wbc_infeasible = 0
        self._sum_pitch_accel_err = 0.0
        self._n_pitch_accel_samples = 0

        # Run until the first MPC-decision point so an observation is ready
        return self._get_obs(), {}

    def _get_obs(self):
        self.mujoco_go2.update_pin_with_mujoco(self.go2)
        d = self.mujoco_go2.data

        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))

        # Gravity vector in body frame
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
        if self.action_channels == "pitch_only":
            action[1] = 0.0
        elif self.action_channels == "height_only":
            action[0] = 0.0
        raw_pitch_corr_deg = float(action[0]) * MAX_PITCH_CORRECTION_DEG
        raw_height_corr_m  = float(action[1]) * MAX_HEIGHT_CORRECTION_M

        # Low-pass filter the APPLIED correction (fix, per external review):
        # u_applied_t = (1-alpha)*u_applied_{t-1} + alpha*u_RL_t. Prevents
        # residual chatter (even within the amplitude bounds) from creating
        # large force discontinuities step-to-step, since these outputs
        # modify MPC targets directly rather than being smoothed by any
        # downstream low-level controller.
        ALPHA = 0.3
        pitch_corr_deg = (1 - ALPHA) * self._applied_correction[0] + ALPHA * raw_pitch_corr_deg
        height_corr_m  = (1 - ALPHA) * self._applied_correction[1] + ALPHA * raw_height_corr_m
        self._applied_correction = np.array([pitch_corr_deg, height_corr_m])
        # Policy observes the actually-applied (smoothed) correction, not
        # the raw action, so it has an accurate view of committed state
        self._last_correction = np.array(
            [pitch_corr_deg / MAX_PITCH_CORRECTION_DEG,
             height_corr_m / MAX_HEIGHT_CORRECTION_M], dtype=np.float32)

        self._n_decision_steps += 1
        if abs(action[0]) > 0.95:
            self._n_raw_pitch_saturated += 1
        if abs(action[1]) > 0.95:
            self._n_raw_height_saturated += 1

        # Run CTRL_DECIM sim sub-steps per RL step, doing exactly one MPC
        # update at the start (RL acts once per MPC update, not every
        # control tick -- matches the natural timescale of terrain
        # adaptation, not fast joint-level control which stays MPC/WBC's job)
        mpc_solve_failed = False
        pitch_accel_pred, pitch_accel_direct = None, None
        for _ in range(STEPS_PER_MPC):
            time_now_s = float(self.mujoco_go2.data.time)
            self.mujoco_go2.update_pin_with_mujoco(self.go2)

            if self._ctrl_i % STEPS_PER_MPC == 0:
                x0_before = self.go2.compute_com_x_vec().reshape(-1)
                # slope_deg passed to generate_traj/WBC is the POLICY'S
                # correction, NOT the true terrain slope -- the policy never
                # observes ground truth, only proprioception
                self.traj.generate_traj(
                    self.go2, self.gait, time_now_s, X_VEL_DES, 0.0,
                    0.27 + height_corr_m, 0.0, time_step=MPC_DT,
                    slope_deg=pitch_corr_deg,
                )
                sol = self.mpc.solve_QP(self.go2, self.traj, False)
                if not self.mpc.last_solve_success:
                    # A failed QP solve (e.g. from an extreme/nonsensical
                    # RL correction during exploration) should not propagate
                    # a garbage force solution into the physics sim -- treat
                    # it as an immediate episode failure instead, same as a
                    # physical fall.
                    mpc_solve_failed = True
                    self._n_mpc_infeasible += 1
                    break
                N = self.traj.N
                w_opt = sol["x"].full().flatten()
                self.U_opt = w_opt[12*N:].reshape((12, N), order="F")

                # Predicted pitch acceleration from the MPC's own model,
                # for the predicted-vs-measured logging requested in review
                x1_pred = self.traj.Ad @ x0_before + self.traj.Bd[0] @ self.U_opt[:, 0] + self.traj.gd.reshape(-1)
                pitch_accel_pred = (x1_pred[10] - x0_before[10]) / MPC_DT

            mpc_force = self.U_opt[:, 0]
            leg_outputs = self.leg_ctrl.compute_all_torques(
                self.go2, self.gait, mpc_force, time_now_s,
                use_wbc=True, slope_deg=pitch_corr_deg,
            )
            if not self.leg_ctrl.last_wbc_success:
                self._n_wbc_infeasible += 1
            tau_raw = np.zeros(12)
            for leg in LEG_NAMES:
                tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
            self._tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
            self._ctrl_i += 1

            for _ in range(CTRL_DECIM):
                mj.mj_step1(self.mujoco_go2.model, self.mujoco_go2.data)
                self.mujoco_go2.set_joint_torque(self._tau_hold)
                mj.mj_step2(self.mujoco_go2.model, self.mujoco_go2.data)
                self._sim_k += 1

        # Measured pitch acceleration, compared against the MPC's own
        # prediction from the start of this decision step (predicted-vs-
        # measured logging per review)
        if pitch_accel_pred is not None and not mpc_solve_failed:
            self.mujoco_go2.update_pin_with_mujoco(self.go2)
            x1_actual = self.go2.compute_com_x_vec().reshape(-1)
            pitch_accel_direct = (x1_actual[10] - x0_before[10]) / (STEPS_PER_MPC * CTRL_DT)
            self._sum_pitch_accel_err += abs(pitch_accel_pred - pitch_accel_direct)
            self._n_pitch_accel_samples += 1

        # --- Reward and termination, using TRUE terrain-relative quantities
        # (fall detection/reward can use ground truth; only the POLICY's
        # observation must stay proprioception-only) ---
        d = self.mujoco_go2.data
        base_x, base_z = d.qpos[0], d.qpos[2]
        slope_rad = np.radians(self._true_slope_deg)
        terrain_h = -base_x * np.tan(slope_rad) if self._terrain_kind == "slope" else 0.0
        height_rel = base_z - terrain_h

        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
        pitch_rel = pitch - slope_rad if self._terrain_kind == "slope" else pitch

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

        truncated = self._ctrl_i >= self.max_ctrl_steps
        if truncated and not terminated:
            term_reason = "success_timeout"

        r_upright = np.exp(-5.0 * (roll**2 + pitch_rel**2))
        r_forward = np.clip(self.go2.current_config.base_vel[0], 0, X_VEL_DES)
        r_correction_penalty = -0.05 * float(np.sum(action**2))  # small: encourage minimal intervention
        r_alive = 0.1

        reward = r_upright + r_forward + r_correction_penalty + r_alive
        if terminated:
            reward -= 10.0

        self._episode_reward += reward
        obs = self._get_obs()
        info = {
            # Per-step diagnostics, always present (per review): raw vs
            # applied correction, MPC/WBC feasibility this step, predicted
            # vs measured pitch acceleration, true terrain angle -- lets a
            # training callback log full per-step detail, not just
            # episode-end summaries.
            "raw_pitch_corr_deg": raw_pitch_corr_deg,
            "applied_pitch_corr_deg": pitch_corr_deg,
            "raw_height_corr_m": raw_height_corr_m,
            "applied_height_corr_m": height_corr_m,
            "mpc_feasible_this_step": not mpc_solve_failed,
            "wbc_feasible_this_step": self.leg_ctrl.last_wbc_success,
            "pitch_accel_pred": pitch_accel_pred,
            "pitch_accel_measured": pitch_accel_direct,
            "true_slope_deg": self._true_slope_deg,
        }
        if terminated or truncated:
            n = max(self._n_decision_steps, 1)
            n_accel = max(self._n_pitch_accel_samples, 1)
            info["residual_episode"] = {
                "r": self._episode_reward, "l": self._ctrl_i, "terrain": self._terrain_kind,
                "true_slope_deg": self._true_slope_deg,
                "term_reason": term_reason,
                # Saturation tracking (per review): if either stays near 100%,
                # the residual range may be too small or the policy may be
                # exploiting the boundary
                "frac_pitch_saturated": self._n_raw_pitch_saturated / n,
                "frac_height_saturated": self._n_raw_height_saturated / n,
                # QP feasibility, tracked distinct from physical falls (per review)
                "n_mpc_infeasible": self._n_mpc_infeasible,
                "n_wbc_infeasible": self._n_wbc_infeasible,
                "frac_mpc_infeasible": self._n_mpc_infeasible / n,
                "frac_wbc_infeasible": self._n_wbc_infeasible / n,
                # Predicted-vs-measured pitch acceleration error, mean over episode
                "mean_pitch_accel_err": self._sum_pitch_accel_err / n_accel,
                # Raw vs applied correction at episode end, for visibility into filtering
                "final_raw_action": action.tolist(),
                "final_applied_correction": self._applied_correction.tolist(),
            }

        return obs, reward, terminated, truncated, info
