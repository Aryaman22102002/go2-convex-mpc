# ex05_multi_gait_benchmark.py. Runs all 4 gaits (trot-in-place, forward, sideway, rotation) with both Baseline (JᵀF) and WBC controllers.


import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from convex_mpc.sim_params import MU, MU_SAFE, get_friction_tag, apply_friction_to_xml, VEL_FORWARD, VEL_SIDEWAY, VEL_ROTATION
apply_friction_to_xml()  # patch XML to match MU in sim_params.py
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.plot_helper import hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Shared parameters
# --------------------------------------------------------------------------------
INITIAL_X_POS    = -5
INITIAL_Y_POS    = 0
RUN_SIM_LENGTH_S = 5.0
RENDER_HZ        = 120.0
RENDER_DT        = 1.0 / RENDER_HZ
REALTIME_FACTOR  = 1
GAIT_HZ          = 3
GAIT_DUTY        = 0.6
GAIT_T           = 1.0 / GAIT_HZ
SIM_HZ           = 1000
SIM_DT           = 1.0 / SIM_HZ
CTRL_HZ          = 200
CTRL_DT          = 1.0 / CTRL_HZ
CTRL_DECIM       = SIM_HZ // CTRL_HZ
SIM_STEPS        = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS       = int(RUN_SIM_LENGTH_S * CTRL_HZ)
MPC_DT           = GAIT_T / 16
MPC_HZ           = 1.0 / MPC_DT
STEPS_PER_MPC    = max(1, int(CTRL_HZ // MPC_HZ))

HIP_LIM  = 23.7
ABD_LIM  = 23.7
KNEE_LIM = 45.43
SAFETY   = 0.9
TAU_LIM  = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
])
LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6),
             "RL": slice(6,9), "RR": slice(9,12)}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{get_friction_tag()}")
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# Gait configurations
# --------------------------------------------------------------------------------
@dataclass
class GaitConfig:
    name:       str
    tag:        str
    x_vel:      float
    y_vel:      float
    z_pos:      float
    yaw_rate:   float

GAITS = [
    GaitConfig("Trot In Place",  "trot_inplace",  0.0,         0.0,         0.27, 0.0),
    GaitConfig("Trot Forward",   "trot_forward",  VEL_FORWARD, 0.0,         0.27, 0.0),
    GaitConfig("Trot Sideway",   "trot_sideway",  0.0,         VEL_SIDEWAY, 0.27, 0.0),
    GaitConfig("Trot Rotation",  "trot_rotation", 0.0,         0.0,         0.27, VEL_ROTATION),
]

# --------------------------------------------------------------------------------
# Core simulation function
# --------------------------------------------------------------------------------
def run_simulation(gait_cfg: GaitConfig, use_wbc: bool):
    label    = "WBC" if use_wbc else "Baseline"
    ctrl_label = f"{gait_cfg.name} | {label}"
    print(f"\n  [{ctrl_label}]")

    @dataclass
    class FootTraj:
        pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
        vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))

    x_vec           = np.zeros((12, CTRL_STEPS))
    mpc_force_world = np.zeros((12, CTRL_STEPS))
    tau_raw         = np.zeros((12, CTRL_STEPS))
    tau_cmd_log     = np.zeros((12, CTRL_STEPS))
    time_log        = np.zeros(CTRL_STEPS)
    foot_traj       = FootTraj()
    fx_viol_log     = np.zeros((4, CTRL_STEPS))
    fy_viol_log     = np.zeros((4, CTRL_STEPS))
    fz_viol_log     = np.zeros((4, CTRL_STEPS))
    tau_viol_log    = np.zeros((4, CTRL_STEPS))
    stance_log      = np.zeros((4, CTRL_STEPS), dtype=bool)
    mpc_solve_ms    = []
    mpc_update_ms   = []
    wbc_solve_ms    = []

    go2        = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    leg_ctrl   = LegController()
    traj       = ComTraj(go2)
    gait       = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    traj.generate_traj(go2, gait, 0.0,
                       gait_cfg.x_vel, gait_cfg.y_vel,
                       gait_cfg.z_pos, gait_cfg.yaw_rate,
                       time_step=MPC_DT)
    mpc   = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    time_log_render, q_log_render, tau_log_render = [], [], []
    next_render_t = 0.0
    ctrl_i    = 0
    tau_hold  = np.zeros(12, dtype=float)

    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_go2.data.time)

        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            mujoco_go2.update_pin_with_mujoco(go2)
            x_vec[:, ctrl_i]  = go2.compute_com_x_vec().reshape(-1)
            time_log[ctrl_i]  = time_now_s

            if (ctrl_i % STEPS_PER_MPC) == 0:
                traj.generate_traj(go2, gait, time_now_s,
                                   gait_cfg.x_vel, gait_cfg.y_vel,
                                   gait_cfg.z_pos, gait_cfg.yaw_rate,
                                   time_step=MPC_DT)
                sol = mpc.solve_QP(go2, traj, False)
                mpc_solve_ms.append(mpc.solve_time)
                mpc_update_ms.append(mpc.update_time)
                N     = traj.N
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12*N:].reshape((12, N), order="F")

            mpc_force_world[:, ctrl_i] = U_opt[:, 0]

            leg_outputs = leg_ctrl.compute_all_torques(
                go2, gait, mpc_force_world[:, ctrl_i],
                time_now_s, use_wbc=use_wbc
            )
            if use_wbc:
                wbc_solve_ms.append(leg_ctrl.wbc_solve_time_ms)

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

    return {
        "label":      label,
        "gait_name":  gait_cfg.name,
        "gait_tag":   gait_cfg.tag,
        "ctrl_i":     ctrl_i,
        "t_vec":      time_log[:ctrl_i],
        "x_vec":      x_vec,
        "mpc_force":  mpc_force_world,
        "tau_cmd":    tau_cmd_log,
        "foot_traj":  foot_traj,
        "fx_viol":    fx_viol_log,
        "fy_viol":    fy_viol_log,
        "fz_viol":    fz_viol_log,
        "tau_viol":   tau_viol_log,
        "stance":     stance_log,
        "mpc_solve_ms":  mpc_solve_ms,
        "mpc_update_ms": mpc_update_ms,
        "wbc_solve_ms":   wbc_solve_ms,
        "render": {
            "time": np.asarray(time_log_render),
            "q":    np.asarray(q_log_render),
            "tau":  np.asarray(tau_log_render),
        },
        "mujoco_ref": mujoco_go2,
    }

def violation_pct(logs, key, leg_idx):
    n  = logs["ctrl_i"]
    sc = int(np.sum(logs["stance"][leg_idx, :n]))
    vc = int(np.sum(logs[key][leg_idx, :n] > 0))
    return 100 * vc / max(sc, 1), vc, sc

def max_fx_viol_pct(logs):
    """Return worst-case fx violation % across all 4 legs."""
    return max(violation_pct(logs, "fx_viol", i)[0] for i in range(4))

# --------------------------------------------------------------------------------
# Video export
# --------------------------------------------------------------------------------
try:
    import imageio
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "imageio[ffmpeg]", "--break-system-packages"], check=True)
    import imageio

def export_video(logs, filename):
    model    = logs["mujoco_ref"].model
    data_rep = mj.MjData(model)
    renderer = mj.Renderer(model, height=720, width=1280)
    q_log    = logs["render"]["q"]
    tau_log  = logs["render"]["tau"]

    base_id = model.body("base_link").id
    cam = mj.MjvCamera()
    cam.type        = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.distance    = 2.5
    cam.elevation   = -20
    cam.azimuth     = 90

    frames = []
    for k in range(len(q_log)):
        data_rep.qpos[:] = q_log[k]
        data_rep.ctrl[:] = tau_log[k]
        mj.mj_forward(model, data_rep)
        renderer.update_scene(data_rep, camera=cam)
        frames.append(renderer.render())

    path = os.path.join(SAVE_DIR, filename)
    imageio.mimsave(path, frames, fps=int(RENDER_HZ))
    print(f"  Video saved: {path}")
    renderer.close()

# --------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------
def plot_gait_comparison(base, wbc, gait_tag):
    n_b, n_w = base["ctrl_i"], wbc["ctrl_i"]
    t_b, t_w = base["t_vec"], wbc["t_vec"]

    # Cumulative violation plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{base['gait_name']} — Cumulative Friction Violations: Baseline vs WBC", fontsize=13)
    axes = axes.flatten()
    for i, leg in enumerate(LEG_NAMES):
        ax = axes[i]
        b_any  = ((base["fx_viol"][i,:n_b]>0)|(base["fy_viol"][i,:n_b]>0)).astype(float)
        w_any  = ((wbc["fx_viol"][i,:n_w]>0) |(wbc["fy_viol"][i,:n_w]>0)).astype(float)
        b_cum  = np.cumsum(b_any)
        w_cum  = np.cumsum(w_any)
        ax.plot(t_b, b_cum, color='tab:red',  lw=1.5, label=f'Baseline  final={int(b_cum[-1])}')
        ax.plot(t_w, w_cum, color='tab:blue', lw=1.5, label=f'WBC       final={int(w_cum[-1])}')
        ax.fill_between(t_b, b_cum, w_cum[:len(t_b)], alpha=0.1, color='tab:red')
        ax.set_title(f'{leg} leg')
        ax.set_ylabel('Cumulative violations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"cumulative_{gait_tag}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

def plot_summary_table(results):
    """Plot summary table of max fx violation % across all gaits."""
    gaits      = [r[0]["gait_name"] for r in results]
    base_vals  = [max_fx_viol_pct(r[0]) for r in results]
    wbc_vals   = [max_fx_viol_pct(r[1]) for r in results]

    x     = np.arange(len(gaits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, base_vals, width, label='Baseline (JᵀF)', color='tab:red',  alpha=0.8)
    ax.bar(x + width/2, wbc_vals,  width, label='WBC',             color='tab:blue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gaits, fontsize=11)
    ax.set_ylabel('Max fx violation rate (% of stance ticks)')
    ax.set_title('Friction Cone Violation Rate: Baseline vs WBC across All Gaits')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (b, w) in enumerate(zip(base_vals, wbc_vals)):
        ax.text(i - width/2, b + 0.3, f'{b:.1f}%', ha='center', fontsize=9, color='tab:red')
        ax.text(i + width/2, w + 0.3, f'{w:.1f}%', ha='center', fontsize=9, color='tab:blue')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "summary_all_gaits.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved summary: {path}")
    plt.show(block=False)


def plot_wbc_timing(all_results):
    """Plot WBC solve time across all gaits."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("WBC QP Solve Time per Control Tick (200 Hz budget = 5.0 ms)", fontsize=13)
    axes = axes.flatten()
    budget = 5.0  # ms at 200 Hz

    for idx, (base, wbc) in enumerate(all_results):
        ax = axes[idx]
        ms = wbc["wbc_solve_ms"]
        if len(ms) == 0:
            continue
        ms = np.array(ms)
        ax.plot(ms, color='tab:blue', lw=0.7, alpha=0.8, label='WBC solve time')
        ax.axhline(budget, color='tab:red', linestyle='--', lw=1.5, label=f'Budget ({budget} ms)')
        ax.axhline(np.mean(ms), color='tab:green', linestyle='-', lw=1.2,
                   label=f'Mean: {np.mean(ms):.2f} ms')
        ax.set_title(wbc["gait_name"])
        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("Control tick")
        ax.set_ylim(0, max(budget * 1.5, np.max(ms) * 1.2))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "wbc_timing.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show(block=False)

def plot_torque_comparison(all_results):
    """
    Plot joint torques baseline vs WBC for the RL leg during trot forward.
    RL leg chosen because it has the highest violation rate in baseline.
    Shows 3 subplots: hip, thigh, calf torques.
    """
    # Find trot forward result
    trot_fwd = next((r for r in all_results if r[0]["gait_tag"] == "trot_forward"), None)
    if trot_fwd is None:
        return
    base, wbc = trot_fwd

    n_b = base["ctrl_i"]
    n_w = wbc["ctrl_i"]
    t_b = base["t_vec"]
    t_w = wbc["t_vec"]

    # RL leg torques: rows 6,7,8 in tau_cmd (hip, thigh, calf)
    RL_HIP   = 6
    RL_THIGH = 7
    RL_CALF  = 8

    joint_names  = ["Hip", "Thigh", "Calf"]
    joint_rows   = [RL_HIP, RL_THIGH, RL_CALF]
    joint_limits = [23.7, 23.7, 45.43]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("RL Leg Joint Torques: Baseline vs WBC (Trot Forward)\n"
                 "Dashed lines show motor limits", fontsize=13)

    colors = {'base': 'tab:red', 'wbc': 'tab:blue'}

    for idx, (ax, jname, jrow, jlim) in enumerate(
            zip(axes, joint_names, joint_rows, joint_limits)):

        tau_b = base["tau_cmd"][jrow, :n_b]
        tau_w = wbc["tau_cmd"][jrow, :n_w]

        ax.plot(t_b, tau_b, color=colors['base'], lw=0.8, alpha=0.9,
                label=f'Baseline (JᵀF)')
        ax.plot(t_w, tau_w, color=colors['wbc'],  lw=0.8, alpha=0.9,
                label=f'WBC')

        ax.axhline( jlim, color='k', linestyle='--', lw=1.0, alpha=0.6,
                   label=f'Limit (+{jlim} Nm)')
        ax.axhline(-jlim, color='k', linestyle='--', lw=1.0, alpha=0.6,
                   label=f'Limit (-{jlim} Nm)')

        # Shade where baseline exceeds limit
        ax.fill_between(t_b,
                        np.where(tau_b >  jlim,  tau_b,  jlim),
                        jlim,  alpha=0.25, color='tab:red')
        ax.fill_between(t_b,
                        np.where(tau_b < -jlim, tau_b, -jlim),
                        -jlim, alpha=0.25, color='tab:red')

        ax.set_ylabel(f'{jname} torque (Nm)')
        ax.legend(loc='upper right', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "torque_comparison_RL_trot_forward.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show(block=False)


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
print("\n" + "="*70)
print("MULTI-GAIT BENCHMARK: Baseline vs WBC")
print("="*70)

all_results = []

for gait_cfg in GAITS:
    print(f"\n{'='*70}")
    print(f"GAIT: {gait_cfg.name}")
    print(f"{'='*70}")

    base_logs = run_simulation(gait_cfg, use_wbc=False)
    wbc_logs  = run_simulation(gait_cfg, use_wbc=True)
    all_results.append((base_logs, wbc_logs))

    plot_gait_comparison(base_logs, wbc_logs, gait_cfg.tag)

    print(f"\n  Exporting videos...")
    export_video(base_logs, f"baseline_{gait_cfg.tag}.mp4")
    export_video(wbc_logs,  f"wbc_{gait_cfg.tag}.mp4")

# Summary table plot
plot_summary_table(all_results)

# Print full results table
print("\n" + "="*70)
print("FULL RESULTS TABLE")
print("="*70)
print(f"{'Gait':<20} {'Controller':<12} {'FL fx%':>8} {'FR fx%':>8} {'RL fx%':>8} {'RR fx%':>8}")
print("-"*70)
for base, wbc in all_results:
    for logs in [base, wbc]:
        n = logs["ctrl_i"]
        vals = [violation_pct(logs, "fx_viol", i)[0] for i in range(4)]
        print(f"{logs['gait_name']:<20} {logs['label']:<12} "
              f"{vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}% {vals[3]:>7.1f}%")
    print()

plot_wbc_timing(all_results)
plot_torque_comparison(all_results)
print(f"\nAll outputs saved to: {SAVE_DIR}")
hold_until_all_fig_closed()
