"""
Demo 02: Trot forward — Baseline vs WBC comparison
Runs the simulation twice and plots side-by-side violation comparison.
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.plot_helper import plot_mpc_result, plot_swing_foot_traj, plot_solve_time, hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

INITIAL_X_POS     = -5
INITIAL_Y_POS     = 0
RUN_SIM_LENGTH_S  = 5.0
RENDER_HZ         = 120.0
RENDER_DT         = 1.0 / RENDER_HZ
REALTIME_FACTOR   = 1

@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float

CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 5.0, 0.8, 0.0, 0.27, 0.0),
]

GAIT_HZ   = 3
GAIT_DUTY = 0.6
GAIT_T    = 1.0 / GAIT_HZ

SIM_HZ   = 1000
SIM_DT   = 1.0 / SIM_HZ
CTRL_HZ  = 200
CTRL_DT  = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ
SIM_STEPS  = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

MPC_DT       = GAIT_T / 16
MPC_HZ       = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

HIP_LIM  = 23.7
ABD_LIM  = 23.7
KNEE_LIM = 45.43
SAFETY   = 0.9

TAU_LIM = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
])

LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}

def get_body_cmd(t):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, 0.27, 0.0

# --------------------------------------------------------------------------------
# Core simulation function
# --------------------------------------------------------------------------------

def run_simulation(use_wbc: bool):
    """Run full simulation. Returns logs dict."""

    label = "WBC" if use_wbc else "Baseline (JᵀF)"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    # Storage
    x_vec        = np.zeros((12, CTRL_STEPS))
    mpc_force_world = np.zeros((12, CTRL_STEPS))
    tau_raw      = np.zeros((12, CTRL_STEPS))
    tau_cmd_log  = np.zeros((12, CTRL_STEPS))
    time_log     = np.zeros(CTRL_STEPS)

    @dataclass
    class FootTraj:
        pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))

    foot_traj = FootTraj()

    fx_viol_log  = np.zeros((4, CTRL_STEPS))
    fy_viol_log  = np.zeros((4, CTRL_STEPS))
    fz_viol_log  = np.zeros((4, CTRL_STEPS))
    tau_viol_log = np.zeros((4, CTRL_STEPS))
    stance_log   = np.zeros((4, CTRL_STEPS), dtype=bool)

    mpc_solve_time_ms  = []
    mpc_update_time_ms = []

    # Init
    go2         = PinGo2Model()
    mujoco_go2  = MuJoCo_GO2_Model()
    leg_ctrl    = LegController()
    traj        = ComTraj(go2)
    gait        = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    x_vel_des_body = y_vel_des_body = 0.0
    z_pos_des_body = 0.27
    yaw_rate_des_body = 0.0

    traj.generate_traj(go2, gait, 0.0,
                       x_vel_des_body, y_vel_des_body,
                       z_pos_des_body, yaw_rate_des_body,
                       time_step=MPC_DT)
    mpc    = CentroidalMPC(go2, traj)
    U_opt  = np.zeros((12, traj.N), dtype=float)

    time_log_render, q_log_render, tau_log_render = [], [], []
    next_render_t = 0.0

    ctrl_i    = 0
    tau_hold  = np.zeros(12, dtype=float)
    sim_start = time.perf_counter()

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)

        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)
            mujoco_go2.update_pin_with_mujoco(go2)
            x_vec[:, ctrl_i] = go2.compute_com_x_vec().reshape(-1)
            time_log[ctrl_i] = time_now_s

            if (ctrl_i % STEPS_PER_MPC) == 0:
                print(f"\r  Sim time: {time_now_s:.3f} s", end="", flush=True)
                traj.generate_traj(go2, gait, time_now_s,
                                   x_vel_des_body, y_vel_des_body,
                                   z_pos_des_body, yaw_rate_des_body,
                                   time_step=MPC_DT)
                sol = mpc.solve_QP(go2, traj, False)
                mpc_solve_time_ms.append(mpc.solve_time)
                mpc_update_time_ms.append(mpc.update_time)
                N      = traj.N
                w_opt  = sol["x"].full().flatten()
                U_opt  = w_opt[12*N:].reshape((12, N), order="F")

            mpc_force_world[:, ctrl_i] = U_opt[:, 0]

            leg_outputs = leg_ctrl.compute_all_torques(
                go2, gait, mpc_force_world[:, ctrl_i], time_now_s,
                use_wbc=use_wbc
            )

            for i, leg in enumerate(LEG_NAMES):
                out = leg_outputs[leg]
                tau_raw[LEG_SLICE[leg], ctrl_i]           = out.tau
                foot_traj.pos_des[LEG_SLICE[leg], ctrl_i] = out.pos_des
                foot_traj.pos_now[LEG_SLICE[leg], ctrl_i] = out.pos_now
                foot_traj.vel_des[LEG_SLICE[leg], ctrl_i] = out.vel_des
                foot_traj.vel_now[LEG_SLICE[leg], ctrl_i] = out.vel_now
                fx_viol_log[i,  ctrl_i] = out.fx_viol
                fy_viol_log[i,  ctrl_i] = out.fy_viol
                fz_viol_log[i,  ctrl_i] = out.fz_viol
                tau_viol_log[i, ctrl_i] = out.tau_viol
                stance_log[i,   ctrl_i] = out.is_stance

            tau_cmd_log[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
            tau_hold = tau_cmd_log[:, ctrl_i].copy()
            ctrl_i += 1

        mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
        mujoco_go2.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

        t_after = float(mujoco_go2.data.time)
        if t_after + 1e-12 >= next_render_t:
            time_log_render.append(t_after)
            q_log_render.append(mujoco_go2.data.qpos.copy())
            tau_log_render.append(tau_hold.copy())
            next_render_t += RENDER_DT

    elapsed = time.perf_counter() - sim_start
    print(f"\n  Done. Elapsed: {elapsed:.2f}s  ctrl_ticks: {ctrl_i}/{CTRL_STEPS}")

    # Print violation summary
    print(f"\n  ===== VIOLATION SUMMARY ({label}) =====")
    for i, name in enumerate(LEG_NAMES):
        sc   = int(np.sum(stance_log[i, :ctrl_i]))
        fxc  = int(np.sum(fx_viol_log[i,  :ctrl_i] > 0))
        fyc  = int(np.sum(fy_viol_log[i,  :ctrl_i] > 0))
        fzc  = int(np.sum(fz_viol_log[i,  :ctrl_i] > 0))
        tauc = int(np.sum(tau_viol_log[i, :ctrl_i] > 0))
        print(f"  {name} ({sc} stance): "
              f"fx={fxc} ({100*fxc/max(sc,1):.1f}%)  "
              f"fy={fyc} ({100*fyc/max(sc,1):.1f}%)  "
              f"fz={fzc} ({100*fzc/max(sc,1):.1f}%)  "
              f"tau={tauc} ({100*tauc/max(sc,1):.1f}%)")

    return {
        "label":         label,
        "ctrl_i":        ctrl_i,
        "t_vec":         time_log[:ctrl_i],
        "x_vec":         x_vec,
        "mpc_force":     mpc_force_world,
        "tau_cmd":       tau_cmd_log,
        "foot_traj":     foot_traj,
        "fx_viol":       fx_viol_log,
        "fy_viol":       fy_viol_log,
        "fz_viol":       fz_viol_log,
        "tau_viol":      tau_viol_log,
        "stance":        stance_log,
        "mpc_solve_ms":  mpc_solve_time_ms,
        "mpc_update_ms": mpc_update_time_ms,
        "render": {
            "time": np.asarray(time_log_render),
            "q":    np.asarray(q_log_render),
            "tau":  np.asarray(tau_log_render),
        },
        "go2_ref":       go2,
        "mujoco_ref":    mujoco_go2,
    }

# --------------------------------------------------------------------------------
# Comparison plots
# --------------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SAVE_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")

def plot_comparison(base, wbc):
    n_base = base["ctrl_i"]
    n_wbc  = wbc["ctrl_i"]
    t_b    = base["t_vec"]
    t_w    = wbc["t_vec"]

    # ------------------------------------------------------------------
    # Plot 1: 4 separate figures, one per leg
    #         Each figure: baseline on top, WBC on bottom
    # ------------------------------------------------------------------
    for i, leg in enumerate(LEG_NAMES):
        fig, (ax_base, ax_wbc) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(f"{leg} Leg — Friction Cone Margin: Baseline vs WBC\n"
                     "Positive = slip risk, Negative = safe", fontsize=13)

        for ax, logs, t_vec, n, label in [
            (ax_base, base, t_b, n_base, base["label"]),
            (ax_wbc,  wbc,  t_w, n_wbc,  wbc["label"]),
        ]:
            ax.plot(t_vec, logs["fx_viol"][i, :n], color='tab:blue',   label='|fx|−μfz', lw=0.9)
            ax.plot(t_vec, logs["fy_viol"][i, :n], color='tab:orange', label='|fy|−μfz', lw=0.9)
            ax.axhline(0, color='k', linestyle='--', lw=1.0, label='violation threshold')
            ax.fill_between(t_vec, np.maximum(logs["fx_viol"][i, :n], 0),
                            alpha=0.3, color='tab:blue')
            ax.fill_between(t_vec, np.maximum(logs["fy_viol"][i, :n], 0),
                            alpha=0.3, color='tab:orange')
            ax.set_ylim(-20, 5)
            ax.set_ylabel('Margin (N)')
            ax.set_title(label)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)

        ax_wbc.set_xlabel('Time (s)')
        plt.tight_layout()
        save_fig(fig, f"friction_margin_{leg}.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Cumulative friction violation count — baseline vs WBC
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig2.suptitle("Cumulative Friction Violations: Baseline vs WBC\n"
                  "(counts violations where |fx| or |fy| > μfz)", fontsize=13)

    axes2 = axes2.flatten()
    for i, leg in enumerate(LEG_NAMES):
        ax = axes2[i]

        base_any   = ((base["fx_viol"][i, :n_base] > 0) |
                      (base["fy_viol"][i, :n_base] > 0)).astype(float)
        wbc_any    = ((wbc["fx_viol"][i, :n_wbc]  > 0) |
                      (wbc["fy_viol"][i, :n_wbc]  > 0)).astype(float)
        base_cumul = np.cumsum(base_any)
        wbc_cumul  = np.cumsum(wbc_any)

        ax.plot(t_b, base_cumul, color='tab:red',  lw=1.5,
                label=f'Baseline (JᵀF)  final={int(base_cumul[-1])}')
        ax.plot(t_w, wbc_cumul,  color='tab:blue', lw=1.5,
                label=f'WBC              final={int(wbc_cumul[-1])}')
        ax.fill_between(t_b, base_cumul, wbc_cumul[:len(t_b)],
                        alpha=0.1, color='tab:red', label='Improvement')
        ax.set_title(f'{leg} leg')
        ax.set_ylabel('Cumulative violations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes2[2:]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    save_fig(fig2, "cumulative_violations.png")
    plt.show(block=False)

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

base_logs = run_simulation(use_wbc=False)
wbc_logs  = run_simulation(use_wbc=True)

plot_comparison(base_logs, wbc_logs)

# Standard plots for BOTH runs
for logs, tag in [(base_logs, "baseline"), (wbc_logs, "wbc")]:
    t_vec = logs["t_vec"]
    plt.figure()
    plot_swing_foot_traj(t_vec, logs["foot_traj"], False)
    plt.savefig(os.path.join(SAVE_DIR, f"foot_trajectories_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    plt.figure()
    plot_mpc_result(t_vec, logs["mpc_force"], logs["tau_cmd"], logs["x_vec"], block=False)
    plt.savefig(os.path.join(SAVE_DIR, f"mpc_result_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    plt.figure()
    plot_solve_time(logs["mpc_solve_ms"], logs["mpc_update_ms"], MPC_DT, MPC_HZ, block=False)
    plt.savefig(os.path.join(SAVE_DIR, f"mpc_timing_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"  Saved standard plots for {tag}")

print(f"\nAll plots saved to: {SAVE_DIR}")

# --------------------------------------------------------------------------------
# Video export
# --------------------------------------------------------------------------------

try:
    import imageio
except ImportError:
    print("Installing imageio...")
    import subprocess
    subprocess.run(["pip", "install", "imageio[ffmpeg]", "--break-system-packages"], check=True)
    import imageio

def export_video(logs, filename, height=720, width=1280):
    model    = logs["mujoco_ref"].model
    data_rep = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)

    q_log   = logs["render"]["q"]
    tau_log = logs["render"]["tau"]
    t_log   = logs["render"]["time"]

    # Set up tracking camera — side view following the robot base
    base_id = model.body("base_link").id
    cam = mj.MjvCamera()
    cam.type       = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.distance   = 2.5
    cam.elevation  = -20
    cam.azimuth    = 90   # side view

    frames = []
    print(f"  Rendering {len(t_log)} frames for {filename}...")

    for k in range(len(t_log)):
        data_rep.qpos[:] = q_log[k]
        data_rep.ctrl[:] = tau_log[k]
        mj.mj_forward(model, data_rep)
        renderer.update_scene(data_rep, camera=cam)
        frame = renderer.render()
        frames.append(frame)

    path = os.path.join(SAVE_DIR, filename)
    imageio.mimsave(path, frames, fps=int(RENDER_HZ))
    print(f"  Saved: {path}")
    renderer.close()

print("\nExporting videos...")
export_video(base_logs, "baseline.mp4")
export_video(wbc_logs,  "wbc.mp4")

print(f"\nDone. All outputs in: {SAVE_DIR}")
hold_until_all_fig_closed()
