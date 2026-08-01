"""
test_mpc_model_consistency.py

Decisive audit, per external review: compares the MPC's own internal
linearized prediction (x_dot = A@x + B@f + g) against the acceleration
computed DIRECTLY from actual world-frame contact geometry, using the
SAME force vector, on slope terrain. If linear acceleration agrees but
angular (especially pitch) acceleration does not, the model error is in
the lever-arm terms of B_d(k) (foothold positions), confirming the
diagnosed mechanism precisely -- and potentially pointing at a much
simpler direct fix than RL (supplying B_d with actual world-frame
footholds) rather than needing a learned residual at all.
"""
import numpy as np
import sys

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.gait import Gait

GAIT_HZ = 3
GAIT_DUTY = 0.6
MPC_DT = (1.0 / GAIT_HZ) / 16


def run_test(xml_path, slope_deg, label, n_warmup_steps=30):
    import mujoco as mj
    from convex_mpc.leg_controller import LegController, LEG_NAMES

    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model(xml_path=xml_path)
    gait = Gait(GAIT_HZ, GAIT_DUTY)
    traj = ComTraj(go2)
    leg_ctrl = LegController()

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = 0.0, 0.0
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = 1.0 / 1000

    traj.generate_traj(go2, gait, 0.0, 0.6, 0.0, 0.27, 0.0,
                        time_step=MPC_DT, slope_deg=slope_deg)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    # Run several REAL control steps first, letting genuine slope-induced
    # state divergence actually develop (fix: comparing at t=0, before the
    # robot has taken any physical step, is trivially uninformative -- the
    # robot's real geometry is identical regardless of slope at that exact
    # instant, since nothing has happened yet to make it differ).
    LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}
    TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
    for step_i in range(n_warmup_steps):
        t_now = step_i * MPC_DT * 4  # approx CTRL_DT-scale advance
        mujoco_go2.update_pin_with_mujoco(go2)
        traj.generate_traj(go2, gait, float(mujoco_go2.data.time), 0.6, 0.0, 0.27, 0.0,
                            time_step=MPC_DT, slope_deg=slope_deg)
        sol = mpc.solve_QP(go2, traj, False)
        if not mpc.last_solve_success:
            print(f"  [warmup] MPC solve failed at step {step_i}, stopping warmup early")
            break
        N = traj.N
        w_opt = sol["x"].full().flatten()
        U_opt = w_opt[12*N:].reshape((12, N), order="F")
        mpc_force = U_opt[:, 0]
        leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force,
                                                     float(mujoco_go2.data.time),
                                                     use_wbc=True, slope_deg=slope_deg)
        tau_raw = np.zeros(12)
        for leg in LEG_NAMES:
            tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
        tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        for _ in range(5):  # CTRL_DECIM equivalent
            mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
            mujoco_go2.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    mujoco_go2.update_pin_with_mujoco(go2)

    # Use a representative synthetic force vector: each stance-ish foot
    # supporting roughly its share of body weight, no lateral force
    m = go2.data.Ig.mass
    g = 9.81
    f_per_foot_z = m * g / 4.0
    f = np.zeros(12)
    for i in range(4):
        f[3*i + 2] = f_per_foot_z

    # --- MPC's own internal prediction, using its stored Ad/Bd(0) ---
    x0 = traj.initial_x_vec.reshape(-1)
    Ad = traj.Ad
    Bd0 = traj.Bd[0]
    gd = traj.gd.reshape(-1)
    x1_pred = Ad @ x0 + Bd0 @ f + gd
    # x = [pos(3), rpy(3), vel(3), omega(3)] -- extract predicted acceleration
    # by finite-differencing against x0 over one MPC_DT step
    accel_pred_linear = (x1_pred[6:9] - x0[6:9]) / MPC_DT
    accel_pred_angular = (x1_pred[9:12] - x0[9:12]) / MPC_DT

    # --- Directly computed acceleration from ACTUAL world-frame contact geometry ---
    I_com_world = go2.data.Ig.inertia
    r_fl, r_fr, r_rl, r_rr = go2.get_foot_lever_world()
    r_list = [r_fl, r_fr, r_rl, r_rr]

    accel_direct_linear = np.array([0.0, 0.0, -g])
    for i in range(4):
        accel_direct_linear += f[3*i:3*i+3] / m

    torque_sum = np.zeros(3)
    for i in range(4):
        torque_sum += np.cross(r_list[i], f[3*i:3*i+3])
    omega_body = go2.current_config.base_ang_vel
    I_inv = np.linalg.inv(I_com_world)
    accel_direct_angular = I_inv @ (torque_sum - np.cross(omega_body, I_com_world @ omega_body))

    print(f"\n=== {label} (slope_deg={slope_deg:.1f}) ===")
    print(f"Linear accel  -- MPC predicted: {np.round(accel_pred_linear, 4)}  "
          f"direct: {np.round(accel_direct_linear, 4)}")
    print(f"  difference: {np.round(accel_pred_linear - accel_direct_linear, 4)}")
    print(f"Angular accel -- MPC predicted: {np.round(accel_pred_angular, 4)}  "
          f"direct: {np.round(accel_direct_angular, 4)}")
    print(f"  difference: {np.round(accel_pred_angular - accel_direct_angular, 4)}  "
          f"(roll, pitch, yaw)")

    lin_err = np.linalg.norm(accel_pred_linear - accel_direct_linear)
    ang_err = np.linalg.norm(accel_pred_angular - accel_direct_angular)
    pitch_err = abs(accel_pred_angular[1] - accel_direct_angular[1])
    print(f"  ||linear error||={lin_err:.4f}  ||angular error||={ang_err:.4f}  "
          f"|pitch error|={pitch_err:.4f}")
    return lin_err, ang_err, pitch_err


if __name__ == "__main__":
    print("Testing model consistency: MPC-predicted vs. directly-computed acceleration\n")
    print("(If linear error is small but angular/pitch error is large, the model")
    print(" mismatch is in the lever-arm terms of B_d -- confirming the foothold-")
    print(" geometry hypothesis specifically, per external review.)")

    results = []
    for slope_deg in [-12.0, -6.0, 6.0, 12.0]:
        # Use a pre-generated slope file if available, else generate one
        from convex_mpc.mpc_terrain_gen import make_terrain_xml
        xml_path = make_terrain_xml("slope", slope_deg=slope_deg)
        r = run_test(xml_path, slope_deg, f"slope_{slope_deg}")
        results.append(r)

    r_flat = run_test(
        "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml",
        0.0, "flat (control)"
    )
    results.append(r_flat)

    print("\n=== Summary ===")
    print(f"{'terrain':>15} {'lin_err':>10} {'ang_err':>10} {'pitch_err':>10}")
    labels = ["-12deg", "-6deg", "6deg", "12deg", "flat"]
    for label, (lin_err, ang_err, pitch_err) in zip(labels, results):
        print(f"{label:>15} {lin_err:10.4f} {ang_err:10.4f} {pitch_err:10.4f}")
