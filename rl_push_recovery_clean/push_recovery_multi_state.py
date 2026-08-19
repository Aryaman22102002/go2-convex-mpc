import sys
import argparse
sys.path.insert(0, "src")
import numpy as np
import mujoco as mj
from stable_baselines3 import PPO
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import Gait
from convex_mpc.inplace_terrain_v3 import get_flat_model

GAIT_HZ, GAIT_DUTY = 3, 0.6
SIM_HZ, CTRL_HZ = 1000, 200
CTRL_DECIM = SIM_HZ // CTRL_HZ
MPC_DT = (1.0 / GAIT_HZ) / 16
STEPS_PER_MPC = 2
TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}
V_X_NOM = 0.6
NOMINAL_HEIGHT = 0.27

POLICY_PATH = "mpc_push_recovery_v1_final.zip"
MAX_VY_CORRECTION = 0.4
MAX_ROLL_CORRECTION_DEG = 10.0
ALPHA = 0.3

BASE_BODY_ID = 1
PUSH_DURATION_S = 0.1

RECOVERY_WINDOW_S = 1.25
WX_TH, VY_TH, ROLL_TH = 56.85, 0.188, 2.08

RETURN_KY = 0.3
RETURN_VMAX = 0.05
RETURN_EPS_Y = 0.03
RETURN_EPS_PSI_DEG = 1.5
YAW_K_PSI = 0.3
YAW_RATE_MAX = np.radians(5.0)
ROLL_TH_RETURN = 5.0
DEBOUNCE_STEPS = 5

STATE_NOMINAL, STATE_RECOVERY, STATE_RETURN = "NOMINAL", "RECOVERY", "RETURN"

PUSH_SCHEDULES = {
    "wide": [(3.0, 150.0), (18.0, -150.0), (33.0, 145.0), (48.0, -145.0)],
    "tight": [(3.0, 150.0), (9.0, -150.0), (15.0, 145.0), (21.0, -145.0)],
}
TOTAL_TIME_S = {"wide": 55.0, "tight": 27.0}

_policy = None


def get_policy():
    global _policy
    if _policy is None:
        _policy = PPO.load(POLICY_PATH, device="cpu")
    return _policy


def run_trial(use_policy, seed, base_schedule, total_time_s, record_trajectory=False):
    rng = np.random.RandomState(seed)
    push_schedule = [
        (t + rng.uniform(-0.15, 0.15), f + rng.uniform(-8, 8) * np.sign(f))
        for t, f in base_schedule
    ]

    model = get_flat_model()
    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model(model=model)
    leg_ctrl = LegController()
    traj = ComTraj(go2)
    gait = Gait(GAIT_HZ, GAIT_DUTY)
    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = 0.0, 0.0
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = 1.0 / SIM_HZ
    yaw_nominal_deg = 0.0

    traj.generate_traj(go2, gait, 0.0, V_X_NOM, 0.0, NOMINAL_HEIGHT, 0.0,
                        time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=0.0)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N))

    applied_correction = np.zeros(2)
    ctrl_i = 0
    push_idx = 0
    state = STATE_NOMINAL
    recovery_until = -1.0
    debounce_count = 0
    peak_roll_overall = 0.0
    fell = False
    qpos_trace = [] if record_trajectory else None

    max_ctrl_i = int(total_time_s * CTRL_HZ)
    policy = get_policy() if use_policy else None

    while ctrl_i < max_ctrl_i:
        t_now = float(mujoco_go2.data.time)

        if push_idx < len(push_schedule):
            push_time, push_force = push_schedule[push_idx]
            if push_time <= t_now < push_time + PUSH_DURATION_S:
                mujoco_go2.data.xfrc_applied[BASE_BODY_ID, 1] = push_force
            else:
                mujoco_go2.data.xfrc_applied[BASE_BODY_ID, :] = 0.0
                if t_now >= push_time + 2.0:
                    push_idx += 1
        else:
            mujoco_go2.data.xfrc_applied[BASE_BODY_ID, :] = 0.0

        mujoco_go2.update_pin_with_mujoco(go2)
        d = mujoco_go2.data
        qw, qx, qy, qz = d.qpos[3:7]
        roll_now = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2)))
        wx_now = np.degrees(go2.current_config.base_ang_vel[0])
        vy_now = go2.current_config.base_vel[1]
        e_y = d.qpos[1]
        peak_roll_overall = max(peak_roll_overall, abs(roll_now))

        v_y_cmd_for_detection = 0.0
        if state == STATE_RETURN:
            v_y_cmd_for_detection = float(np.clip(-RETURN_KY * e_y, -RETURN_VMAX, RETURN_VMAX))
        vy_error = vy_now - v_y_cmd_for_detection

        if use_policy:
            if state == STATE_NOMINAL:
                if abs(wx_now) > WX_TH or abs(vy_error) > VY_TH or abs(roll_now) > ROLL_TH:
                    state = STATE_RECOVERY
                    recovery_until = t_now + RECOVERY_WINDOW_S
            elif state == STATE_RECOVERY:
                if t_now > recovery_until:
                    state = STATE_RETURN
            elif state == STATE_RETURN:
                yaw_now_deg = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
                e_psi_deg = yaw_now_deg - yaw_nominal_deg
                if abs(e_y) < RETURN_EPS_Y and abs(e_psi_deg) < RETURN_EPS_PSI_DEG:
                    state = STATE_NOMINAL
                else:
                    disturbance_now = (abs(wx_now) > WX_TH or abs(vy_error) > VY_TH
                                        or abs(roll_now) > ROLL_TH_RETURN)
                    debounce_count = debounce_count + 1 if disturbance_now else 0
                    if debounce_count >= DEBOUNCE_STEPS:
                        state = STATE_RECOVERY
                        recovery_until = t_now + RECOVERY_WINDOW_S
                        debounce_count = 0

        if ctrl_i % STEPS_PER_MPC == 0:
            g_world = np.array([0, 0, -1.0])
            g_body = go2.R_world_to_body @ g_world
            pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
            v_body = go2.current_config.base_vel
            w_body = go2.current_config.base_ang_vel
            current_mask = gait.compute_current_mask(t_now).reshape(4,).astype(np.float32)
            last_corr = np.array([
                applied_correction[0] / MAX_VY_CORRECTION,
                applied_correction[1] / MAX_ROLL_CORRECTION_DEG,
            ], dtype=np.float32)
            roll_rad = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
            obs = np.concatenate([g_body, [roll_rad, pitch], v_body, w_body,
                                   current_mask, last_corr]).astype(np.float32)

            if use_policy and state == STATE_RECOVERY:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)

            raw = np.array([action[0] * MAX_VY_CORRECTION, action[1] * MAX_ROLL_CORRECTION_DEG])
            applied_correction = (1 - ALPHA) * applied_correction + ALPHA * raw

            v_y_cmd = applied_correction[0]
            roll_ref_deg = applied_correction[1]
            yaw_rate_cmd = 0.0

            if use_policy and state == STATE_RETURN:
                v_y_world_des = float(np.clip(-RETURN_KY * e_y, -RETURN_VMAX, RETURN_VMAX))
                v_world_des = np.array([V_X_NOM, v_y_world_des])
                yaw_rad_now = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
                c, s = np.cos(yaw_rad_now), np.sin(yaw_rad_now)
                v_body_des = np.array([
                    c * v_world_des[0] + s * v_world_des[1],
                    -s * v_world_des[0] + c * v_world_des[1],
                ])
                v_y_cmd = float(v_body_des[1])

                yaw_now_deg2 = np.degrees(yaw_rad_now)
                e_psi_deg2 = yaw_now_deg2 - yaw_nominal_deg
                yaw_rate_cmd = float(np.clip(-YAW_K_PSI * np.radians(e_psi_deg2),
                                              -YAW_RATE_MAX, YAW_RATE_MAX))

            traj.generate_traj(go2, gait, t_now, V_X_NOM, v_y_cmd, NOMINAL_HEIGHT, yaw_rate_cmd,
                                time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=roll_ref_deg)
            sol = mpc.solve_QP(go2, traj, False)
            N = traj.N
            U_opt = sol["x"].full().flatten()[12 * N:].reshape((12, N), order="F")

        f = U_opt[:, 0]
        outs = leg_ctrl.compute_all_torques(go2, gait, f, t_now, use_wbc=True, slope_deg=0.0)
        tau = np.zeros(12)
        for leg in LEG_SLICE:
            tau[LEG_SLICE[leg]] = outs[leg].tau
        tau_hold = np.clip(tau, -TAU_LIM, TAU_LIM)

        for _ in range(CTRL_DECIM):
            mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
            mujoco_go2.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

        ctrl_i += 1
        if record_trajectory:
            qpos_trace.append(mujoco_go2.data.qpos.copy())

        if d.qpos[2] < 0.15:
            fell = True
            break

    result = {
        "survived": not fell,
        "peak_roll_deg": peak_roll_overall,
        "duration_s": ctrl_i / CTRL_HZ,
    }
    if record_trajectory:
        result["qpos"] = np.array(qpos_trace)
    return result


def run_randomized_table(spacing, n_trials=10):
    base_schedule = PUSH_SCHEDULES[spacing]
    total_time_s = TOTAL_TIME_S[spacing]

    print(f"\n=== {spacing} spacing, n={n_trials} randomized trials ===")
    print(f"{'mode':>10} {'trial':>6} {'survived':>9} {'peak_roll':>10} {'duration':>9}")
    for mode, use_policy in [("nominal", False), ("RL-3state", True)]:
        survivals, peak_rolls = [], []
        for trial in range(n_trials):
            r = run_trial(use_policy, seed=trial, base_schedule=base_schedule, total_time_s=total_time_s)
            survivals.append(r["survived"])
            peak_rolls.append(r["peak_roll_deg"])
            print(f"{mode:>10} {trial:6d} {str(r['survived']):>9} {r['peak_roll_deg']:10.2f} {r['duration_s']:9.1f}")
        print(f"  {mode} SUMMARY: survival_rate={np.mean(survivals):.1%}  "
              f"mean_peak_roll={np.mean(peak_rolls):.2f}+/-{np.std(peak_rolls):.2f}")


def record_trial_for_video(spacing, seed):
    base_schedule = PUSH_SCHEDULES[spacing]
    total_time_s = TOTAL_TIME_S[spacing]

    for name, use_policy in [("nominal", False), ("rl", True)]:
        r = run_trial(use_policy, seed=seed, base_schedule=base_schedule,
                       total_time_s=total_time_s, record_trajectory=True)
        out_path = f"traj_trial{seed}_{name}.npz"
        np.savez(out_path, qpos=r["qpos"])
        print(f"{name}: survived={r['survived']} peak_roll={r['peak_roll_deg']:.2f} "
              f"duration={r['duration_s']:.1f}s -> saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", choices=["wide", "tight"], default="tight")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--record-trial", type=int, default=None,
                         help="If set, records this trial's trajectory instead of running the full table.")
    args = parser.parse_args()

    if args.record_trial is not None:
        record_trial_for_video(args.spacing, args.record_trial)
    else:
        run_randomized_table(args.spacing, args.n_trials)
