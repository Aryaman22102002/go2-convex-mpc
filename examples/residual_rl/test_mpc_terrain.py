"""
Tests the existing MPC+WBC controller (unmodified, use_wbc=True) on flat,
slope, and step terrain -- the same terrain types used to evaluate the RL
policy -- to determine whether the classical controller already handles
what the RL policy currently cannot (slopes), and to check whether it
handles flat/step at least as well.
"""
import os
os.environ["MPLBACKEND"] = "Agg"  # no interactive plotting needed for this test
import sys
import numpy as np
import mujoco as mj

sys.path.insert(0, '.')
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.mpc_terrain_gen import make_terrain_xml

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


def run_trial(xml_path, terrain_kind, slope_deg, run_length_s, x_vel_des=0.6):
    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model(xml_path=xml_path)
    leg_ctrl = LegController()
    traj = ComTraj(go2)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = 0.0, 0.0  # spawn at world origin (correct local
                                       # ground height regardless of slope
                                       # angle -- same convention as RL terrain)
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    x_vel_des_body, y_vel_des_body = x_vel_des, 0.0
    z_pos_des_body = 0.27
    yaw_rate_des_body = 0.0

    traj.generate_traj(go2, gait, 0.0, x_vel_des_body, y_vel_des_body,
                        z_pos_des_body, yaw_rate_des_body, time_step=MPC_DT,
                        slope_deg=slope_deg)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    SIM_STEPS = int(run_length_s * SIM_HZ)
    CTRL_STEPS = int(run_length_s * CTRL_HZ)

    ctrl_i = 0
    tau_hold = np.zeros(12, dtype=float)
    fell = False
    fall_reason = None
    fall_step = None
    slope_rad = np.radians(slope_deg)
    trace = []

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)

        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            mujoco_go2.update_pin_with_mujoco(go2)

            # --- Fall detection, terrain-relative for slope (same fix
            # applied to the RL evaluation, to avoid unfairly penalizing
            # the controller for a naive world-frame height check while
            # it's walked away from the origin on a tilted floor) ---
            base_x = mujoco_go2.data.qpos[0]
            base_z = mujoco_go2.data.qpos[2]
            terrain_h = -base_x * np.tan(slope_rad) if terrain_kind == "slope" else 0.0
            height_rel = base_z - terrain_h

            qw, qx, qy, qz = mujoco_go2.data.qpos[3:7]
            roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
            pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
            pitch_rel = pitch - slope_rad if terrain_kind == "slope" else pitch

            if ctrl_i < 40:
                trace.append((ctrl_i, base_x, height_rel, np.degrees(roll), np.degrees(pitch_rel)))

            if height_rel < 0.15:
                fell, fall_reason, fall_step = True, "low_height", ctrl_i
                break
            if abs(roll) > 0.8 or abs(pitch_rel) > 0.8:
                fell, fall_reason, fall_step = True, "extreme_tilt", ctrl_i
                break

            if (ctrl_i % STEPS_PER_MPC) == 0:
                traj.generate_traj(go2, gait, time_now_s, x_vel_des_body,
                                    y_vel_des_body, z_pos_des_body,
                                    yaw_rate_des_body, time_step=MPC_DT,
                                    slope_deg=slope_deg)
                sol = mpc.solve_QP(go2, traj, False)
                N = traj.N
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12*N:].reshape((12, N), order="F")

            mpc_force = U_opt[:, 0]
            leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s,
                                                        use_wbc=True, slope_deg=slope_deg)

            tau_raw = np.zeros(12)
            for leg in LEG_NAMES:
                tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
            tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
            ctrl_i += 1

        mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
        mujoco_go2.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    return {
        "fell": fell, "fall_reason": fall_reason, "fall_step": fall_step,
        "trace": trace,
        "ctrl_i": ctrl_i, "final_x": float(mujoco_go2.data.qpos[0]),
    }


if __name__ == "__main__":
    terrain_kind = sys.argv[1]  # "flat", "slope", "step_up", "step_down"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results = []
    for i in range(n_trials):
        if terrain_kind == "flat":
            xml_path = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml"
            slope_deg = 0.0
        elif terrain_kind == "slope":
            slope_deg = np.random.uniform(-15, 15)
            if abs(slope_deg) < 2.0:
                slope_deg = np.sign(slope_deg) * 2.0 if slope_deg != 0 else 5.0
            xml_path = make_terrain_xml("slope", slope_deg=slope_deg)
        elif terrain_kind in ("step_up", "step_down"):
            step_height = np.random.uniform(0.03, 0.08)
            xml_path = make_terrain_xml(terrain_kind, step_height=step_height)
            slope_deg = 0.0
        else:
            raise ValueError(terrain_kind)

        result = run_trial(xml_path, "slope" if terrain_kind == "slope" else terrain_kind,
                            slope_deg, run_length_s=6.0)
        tag = f"slope_deg={slope_deg:.1f}" if terrain_kind == "slope" else terrain_kind
        print(f"Trial {i} ({tag}): fell={result['fell']}  reason={result['fall_reason']}  "
              f"ctrl_steps={result['ctrl_i']}  final_x={result['final_x']:.3f}")
        if terrain_kind == "slope" and result["trace"]:
            print(f"  {'step':>5} {'x':>7} {'height_rel':>11} {'roll_deg':>9} {'pitch_rel_deg':>13}")
            for row in result["trace"]:
                ci, bx, hr, rd, prd = row
                print(f"  {ci:5d} {bx:7.3f} {hr:11.4f} {rd:9.2f} {prd:13.2f}")
        results.append(result)

        if xml_path != "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml":
            try:
                os.unlink(xml_path)
            except Exception:
                pass

    n_survived = sum(1 for r in results if not r["fell"])
    print(f"\n=== {terrain_kind}: {n_survived}/{n_trials} survived full 6.0s trial ===")
