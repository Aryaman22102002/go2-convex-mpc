"""
export_faulty_test_video.py

Renders the SAME trial that test_trained_policy_steep_uphill.py's "zero"
condition reported as "10/10 survived" at -15deg downhill -- but that
script's fall-detection uses ABSOLUTE (not terrain-relative) height/pitch
thresholds, which are blind to real failures on sloped terrain. This
renders the actual trajectory so we can see with our own eyes whether the
robot is genuinely walking normally, or visibly sliding/struggling while
the broken check reports false success.
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
RENDER_HZ = 30.0

slope_deg = float(sys.argv[1]) if len(sys.argv) > 1 else -15.0
out_name = sys.argv[2] if len(sys.argv) > 2 else f"faulty_test_slope{slope_deg:.0f}.mp4"

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

# NOTE: matching the "faulty test" exactly -- slope_deg=0.0 passed
# throughout (assumes flat), no correction ever applied, no
# terrain-relative fall check either (we just render frames, no fall
# check needed for a video export).
traj.generate_traj(go2, gait, 0.0, X_VEL_DES, 0.0, 0.27, 0.0, time_step=MPC_DT, slope_deg=0.0)
mpc = CentroidalMPC(go2, traj)
U_opt = np.zeros((12, traj.N), dtype=float)

SIM_STEPS = int(RUN_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_LENGTH_S * CTRL_HZ)
ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

q_log, t_log = [], []
next_render_t = 0.0
RENDER_DT = 1.0 / RENDER_HZ

print(f"Running 'faulty test' condition at slope_deg={slope_deg} for {RUN_LENGTH_S}s...")
for k in range(SIM_STEPS):
    time_now_s = float(mujoco_go2.data.time)
    if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
        mujoco_go2.update_pin_with_mujoco(go2)

        if ctrl_i % STEPS_PER_MPC == 0:
            traj.generate_traj(go2, gait, time_now_s, X_VEL_DES, 0.0, 0.27, 0.0,
                                time_step=MPC_DT, slope_deg=0.0)
            sol = mpc.solve_QP(go2, traj, False)
            N = traj.N
            U_opt = sol["x"].full().flatten()[12*N:].reshape((12, N), order="F")

        mpc_force = U_opt[:, 0]
        leg_outputs = leg_ctrl.compute_all_torques(go2, gait, mpc_force, time_now_s,
                                                    use_wbc=True, slope_deg=0.0)
        tau_raw = np.zeros(12)
        for leg in LEG_NAMES:
            tau_raw[LEG_SLICE[leg]] = leg_outputs[leg].tau
        tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        ctrl_i += 1

    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_joint_torque(tau_hold)
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    t_after = float(mujoco_go2.data.time)
    if t_after + 1e-9 >= next_render_t:
        q_log.append(mujoco_go2.data.qpos.copy())
        t_log.append(t_after)
        next_render_t += RENDER_DT

print(f"Final x={mujoco_go2.data.qpos[0]:.3f}, z={mujoco_go2.data.qpos[2]:.3f}, "
      f"rendering {len(q_log)} frames...")

renderer = mj.Renderer(mujoco_go2.model, height=720, width=1280)
data_replay = mj.MjData(mujoco_go2.model)
base_id = mujoco_go2.model.body("base_link").id
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = base_id
cam.distance = 2.0
cam.elevation = -15
cam.azimuth = 90

frames = []
for q in q_log:
    data_replay.qpos[:] = q
    mj.mj_forward(mujoco_go2.model, data_replay)
    renderer.update_scene(data_replay, camera=cam)
    frames.append(renderer.render())

import imageio
imageio.mimsave(out_name, frames, fps=int(RENDER_HZ))
print(f"Saved video to {out_name}")
renderer.close()
