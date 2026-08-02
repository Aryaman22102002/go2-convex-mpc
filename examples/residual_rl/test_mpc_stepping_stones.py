"""
test_mpc_stepping_stones.py

Tests the EXISTING, UNMODIFIED MPC+WBC controller directly on stepping-
stone terrain, before building any RL environment around it. Confirms
or refutes the hypothesis that the fixed swing-foot touchdown heuristic
(gait.py) has no mechanism to avoid a gap, by direct measurement rather
than assumption.
"""
import numpy as np
import mujoco as mj

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.inplace_terrain import get_stepping_stones_model, set_stepping_stones

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


def run_trial(gap_width, stone_length=0.3, stone_width=0.6, seed=0):
    model = get_stepping_stones_model()
    rng = np.random.default_rng(seed)
    set_stepping_stones(model, stone_length=stone_length, gap_width=gap_width,
                         stone_width=stone_width)

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
    fell, fall_reason = False, None

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            mujoco_go2.update_pin_with_mujoco(go2)

            base_z = mujoco_go2.data.qpos[2]
            qw, qx, qy, qz = mujoco_go2.data.qpos[3:7]
            roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
            pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))

            if base_z < 0.15:
                fell, fall_reason = True, "low_height"
                break
            if abs(roll) > 0.8 or abs(pitch) > 0.8:
                fell, fall_reason = True, "extreme_tilt"
                break

            if ctrl_i % STEPS_PER_MPC == 0:
                traj.generate_traj(go2, gait, time_now_s, X_VEL_DES, 0.0, 0.27, 0.0, time_step=MPC_DT)
                sol = mpc.solve_QP(go2, traj, False)
                if not mpc.last_solve_success:
                    fell, fall_reason = True, "mpc_infeasible"
                    break
                N = traj.N
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12*N:].reshape((12, N), order="F")

            mpc_force = U_opt[:, 0]
            leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s, use_wbc=True)
            tau_raw = np.zeros(12)
            for leg in LEG_NAMES:
                tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
            tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
            ctrl_i += 1

            for _ in range(CTRL_DECIM):
                mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                mujoco_go2.set_joint_torque(tau_hold)
                mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    return {"fell": fell, "fall_reason": fall_reason, "ctrl_steps": ctrl_i,
            "final_x": float(mujoco_go2.data.qpos[0])}


if __name__ == "__main__":
    import sys
    gap_widths = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    for gap_width in gap_widths:
        results = []
        for trial in range(n_trials):
            r = run_trial(gap_width=gap_width, seed=trial)
            results.append(r)
            print(f"gap={gap_width:.2f}m trial {trial}: fell={r['fell']} "
                  f"reason={r['fall_reason']} steps={r['ctrl_steps']} final_x={r['final_x']:.3f}")
        survived = sum(1 for r in results if not r["fell"])
        print(f"=== gap={gap_width:.2f}m: {survived}/{n_trials} survived full {RUN_LENGTH_S}s ===\n")
