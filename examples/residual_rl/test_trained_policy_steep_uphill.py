"""
test_trained_policy_steep_uphill.py

Directly tests the trained pitch-only policy at the SPECIFIC steep uphill
angles (5, 8, 11, 15 deg) where the dedicated n=50 sweep already proved
zero-residual fails completely (0/10 at every one of these angles). This
is the actual question tonight's slope-weighted retrain needs to answer --
random uniform(-15,15) sampling in eval scripts keeps landing on shallow
angles by chance and never actually tests this.
"""
import sys
import numpy as np
import mujoco as mj
from stable_baselines3 import PPO

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.inplace_terrain import get_universal_model, set_slope

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
X_VEL_DES = 0.6
RUN_LENGTH_S = 6.0
MAX_PITCH_CORRECTION_DEG = 15.0
ALPHA = 0.3


def run_trial(slope_deg, policy=None):
    model = get_universal_model()
    set_slope(model, slope_deg)

    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model(model=model)
    leg_ctrl = LegController()
    traj = ComTraj(go2)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = 0.0, 0.0
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    applied_pitch_corr = 0.0
    last_correction = np.zeros(1, dtype=np.float32)

    # NOTE: policy trained WITHOUT true slope_deg given to it -- reference
    # starts flat (slope_deg=0 passed to generate_traj), policy must infer
    # and correct purely from proprioception, matching training exactly
    traj.generate_traj(go2, gait, 0.0, X_VEL_DES, 0.0, 0.27, 0.0, time_step=MPC_DT, slope_deg=0.0)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    SIM_STEPS = int(RUN_LENGTH_S * SIM_HZ)
    CTRL_STEPS = int(RUN_LENGTH_S * CTRL_HZ)
    ctrl_i = 0
    tau_hold = np.zeros(12, dtype=float)

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            mujoco_go2.update_pin_with_mujoco(go2)
            d = mujoco_go2.data
            base_z = d.qpos[2]
            qw, qx, qy, qz = d.qpos[3:7]
            roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
            pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
            if base_z < 0.15:
                return {"fell": True, "reason": "low_height", "steps": ctrl_i}
            if abs(roll) > 0.8 or abs(pitch) > 0.8:
                return {"fell": True, "reason": "extreme_tilt", "steps": ctrl_i}

            if policy is not None:
                R = go2.R_world_to_body
                g_body = R @ np.array([0, 0, -1.0])
                v_body = go2.current_config.base_vel
                w_body = go2.current_config.base_ang_vel
                current_mask = gait.compute_current_mask(time_now_s).reshape(4,).astype(np.float32)
                obs = np.concatenate([
                    g_body, [roll, pitch], v_body, w_body, current_mask, last_correction
                ]).astype(np.float32)
                action, _ = policy.predict(obs, deterministic=True)
                action = np.clip(action, -1.0, 1.0)
                raw_pitch_corr_deg = float(action[0]) * MAX_PITCH_CORRECTION_DEG
                applied_pitch_corr = (1 - ALPHA) * applied_pitch_corr + ALPHA * raw_pitch_corr_deg
                last_correction = np.array([applied_pitch_corr / MAX_PITCH_CORRECTION_DEG], dtype=np.float32)

            if ctrl_i % STEPS_PER_MPC == 0:
                traj.generate_traj(go2, gait, time_now_s, X_VEL_DES, 0.0, 0.27, 0.0,
                                    time_step=MPC_DT, slope_deg=applied_pitch_corr)
                sol = mpc.solve_QP(go2, traj, False)
                if not mpc.last_solve_success:
                    return {"fell": True, "reason": "mpc_infeasible", "steps": ctrl_i}
                N = traj.N
                U_opt = sol["x"].full().flatten()[12*N:].reshape((12, N), order="F")

            mpc_force = U_opt[:, 0]
            leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s,
                                                        use_wbc=True, slope_deg=applied_pitch_corr)
            tau_raw = np.zeros(12)
            for leg in LEG_NAMES:
                tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
            tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
            ctrl_i += 1

        mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
        mujoco_go2.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    return {"fell": False, "reason": None, "steps": ctrl_i}


if __name__ == "__main__":
    policy_path = sys.argv[1] if len(sys.argv) > 1 else "mpc_residual_pitch_only.zip"
    n_per_angle = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    slope_angles = [5.0, 8.0, 11.0, 15.0]  # exactly where zero-residual already known to fail

    policy = PPO.load(policy_path, device="cpu")

    for slope_deg in slope_angles:
        for condition, pol in [("zero", None), ("trained", policy)]:
            results = [run_trial(slope_deg, policy=pol) for _ in range(n_per_angle)]
            survived = sum(1 for r in results if not r["fell"])
            print(f"slope={slope_deg:+.1f}deg [{condition:8s}]: {survived}/{n_per_angle} survived")
