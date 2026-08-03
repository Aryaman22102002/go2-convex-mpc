"""
test_mpc_genuine_downhill.py

Corrected version of test_mpc_uphill_only.py: uses the REAL environment's
own terrain-relative fall-check formula (height_rel = base_z - terrain_h,
pitch_rel = pitch - slope_rad), not the broken absolute-threshold check
that contaminated earlier standalone tests tonight.

Tests genuine downhill (positive slope_deg, per tonight's corrected
labeling -- ground FALLS as the robot walks forward for positive
slope_deg, confirmed via terrain_h trace + visual side-view video).
"""
import sys
import numpy as np
import mujoco as mj

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
CTRL_HZ = 200
CTRL_DECIM = SIM_HZ // CTRL_HZ
MPC_DT = GAIT_T / 16
STEPS_PER_MPC = max(1, int(CTRL_HZ * MPC_DT))
TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}
RUN_LENGTH_S = 6.0
X_VEL_DES = 0.6


def run_trial(slope_deg, informed=False):
    """informed=True gives the controller the TRUE slope_deg (matching
    zero-residual's terrain-relative fixes); informed=False assumes flat
    throughout (matching what a residual policy corrects FROM)."""
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
    mujoco_go2.model.opt.timestep = 1.0 / SIM_HZ

    ctrl_slope = slope_deg if informed else 0.0
    traj.generate_traj(go2, gait, 0.0, X_VEL_DES, 0.0, 0.27, 0.0,
                        time_step=MPC_DT, slope_deg=ctrl_slope)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    SIM_STEPS = int(RUN_LENGTH_S * SIM_HZ)
    CTRL_STEPS = int(RUN_LENGTH_S * CTRL_HZ)
    ctrl_i = 0
    tau_hold = np.zeros(12, dtype=float)
    slope_rad = np.radians(slope_deg)  # TRUE slope, for the fall-check only

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            mujoco_go2.update_pin_with_mujoco(go2)
            d = mujoco_go2.data
            base_x, base_z = d.qpos[0], d.qpos[2]
            # REAL environment's own terrain-relative formula
            terrain_h = -base_x * np.tan(slope_rad)
            height_rel = base_z - terrain_h
            qw, qx, qy, qz = d.qpos[3:7]
            roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
            pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
            pitch_rel = pitch - slope_rad
            if height_rel < 0.15:
                return {"fell": True, "reason": "low_height", "steps": ctrl_i}
            if abs(roll) > 0.8:
                return {"fell": True, "reason": "extreme_roll", "steps": ctrl_i}
            if abs(pitch_rel) > 0.8:
                return {"fell": True, "reason": "extreme_pitch", "steps": ctrl_i}

            if ctrl_i % STEPS_PER_MPC == 0:
                traj.generate_traj(go2, gait, time_now_s, X_VEL_DES, 0.0, 0.27, 0.0,
                                    time_step=MPC_DT, slope_deg=ctrl_slope)
                sol = mpc.solve_QP(go2, traj, False)
                if not mpc.last_solve_success:
                    return {"fell": True, "reason": "mpc_infeasible", "steps": ctrl_i}
                N = traj.N
                U_opt = sol["x"].full().flatten()[12*N:].reshape((12, N), order="F")

            mpc_force = U_opt[:, 0]
            leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s,
                                                        use_wbc=True, slope_deg=ctrl_slope)
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
    n_per_angle = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    informed = "--informed" in sys.argv
    # POSITIVE slope_deg = genuine downhill (ground falls as robot walks
    # forward), per tonight's corrected labeling
    slope_angles = [2.0, 5.0, 8.0, 11.0, 15.0]

    print(f"Testing genuine DOWNHILL, informed={informed}")
    total_survived, total_trials = 0, 0
    for slope_deg in slope_angles:
        results = [run_trial(slope_deg, informed=informed) for _ in range(n_per_angle)]
        survived = sum(1 for r in results if not r["fell"])
        total_survived += survived
        total_trials += n_per_angle
        reasons = {}
        for r in results:
            if r["fell"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"slope={slope_deg:+.1f}deg: {survived}/{n_per_angle} survived  "
              f"(failures: {reasons})")

    print(f"\n=== GENUINE DOWNHILL OVERALL (informed={informed}): "
          f"{total_survived}/{total_trials} ({total_survived/total_trials:.1%}) ===")
