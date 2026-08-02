"""
test_mpc_flat_step_only.py

Larger, dedicated test of pure MPC+WBC (NO residual) on flat and step
terrain, using the corrected in-place mutation system (inplace_terrain.py),
now that the old file-based test_mpc_terrain.py is known to have a
degrees-vs-radians bug in its slope code path. Flat/step don't use euler
angles at all (pos/size only), so they were never affected by that specific
bug -- but given the old script's overall reliability is now in question,
this gets a clean, larger-n confirmation on the known-correct system.
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
from convex_mpc.inplace_terrain import get_universal_model, set_flat, set_step

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


def run_trial(terrain_kind, step_height=0.0, direction="up"):
    model = get_universal_model()
    if terrain_kind == "flat":
        set_flat(model)
    else:
        set_step(model, step_height, direction=direction)

    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model(model=model)
    leg_ctrl = LegController()
    traj = ComTraj(go2)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = 0.0, 0.0
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    traj.generate_traj(go2, gait, 0.0, X_VEL_DES, 0.0, 0.27, 0.0, time_step=MPC_DT)
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
            base_z = mujoco_go2.data.qpos[2]
            qw, qx, qy, qz = mujoco_go2.data.qpos[3:7]
            roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
            pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
            if base_z < 0.15:
                return {"fell": True, "reason": "low_height", "steps": ctrl_i}
            if abs(roll) > 0.8 or abs(pitch) > 0.8:
                return {"fell": True, "reason": "extreme_tilt", "steps": ctrl_i}

            if ctrl_i % STEPS_PER_MPC == 0:
                traj.generate_traj(go2, gait, time_now_s, X_VEL_DES, 0.0, 0.27, 0.0, time_step=MPC_DT)
                sol = mpc.solve_QP(go2, traj, False)
                if not mpc.last_solve_success:
                    return {"fell": True, "reason": "mpc_infeasible", "steps": ctrl_i}
                N = traj.N
                U_opt = sol["x"].full().flatten()[12*N:].reshape((12, N), order="F")

            mpc_force = U_opt[:, 0]
            leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s, use_wbc=True)
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
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    step_heights = [0.03, 0.05, 0.06, 0.08]  # matches training range

    print(f"=== FLAT ({n_trials} trials) ===")
    results = [run_trial("flat") for _ in range(n_trials)]
    survived = sum(1 for r in results if not r["fell"])
    for i, r in enumerate(results):
        print(f"trial {i}: fell={r['fell']} reason={r['reason']} steps={r['steps']}")
    print(f"=== FLAT: {survived}/{n_trials} ({survived/n_trials:.1%}) survived ===\n")

    for direction in ["up", "down"]:
        total_survived, total_trials = 0, 0
        print(f"=== STEP_{direction.upper()} ({n_trials} trials per height) ===")
        for h in step_heights:
            results = [run_trial("step", step_height=h, direction=direction) for _ in range(n_trials)]
            survived = sum(1 for r in results if not r["fell"])
            total_survived += survived
            total_trials += n_trials
            for i, r in enumerate(results):
                print(f"height={h:.2f}m trial {i}: fell={r['fell']} reason={r['reason']} steps={r['steps']}")
            print(f"=== step_{direction} height={h:.2f}m: {survived}/{n_trials} survived ===\n")
        print(f"=== STEP_{direction.upper()} OVERALL: {total_survived}/{total_trials} "
              f"({total_survived/total_trials:.1%}) survived ===\n")
