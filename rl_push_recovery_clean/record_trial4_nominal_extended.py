import sys
sys.path.insert(0, 'src')
import numpy as np
import mujoco as mj
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController, LEG_NAMES
from convex_mpc.gait import Gait
from convex_mpc.inplace_terrain_v3 import get_flat_model

GAIT_HZ, GAIT_DUTY = 3, 0.6
SIM_HZ, CTRL_HZ = 1000, 200
CTRL_DECIM = SIM_HZ // CTRL_HZ
MPC_DT = (1.0/GAIT_HZ) / 16
STEPS_PER_MPC = 2
TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
LEG_SLICE = {"FL": slice(0,3), "FR": slice(3,6), "RL": slice(6,9), "RR": slice(9,12)}
V_X_NOM = 0.6
NOMINAL_HEIGHT = 0.27
BASE_BODY_ID = 1
PUSH_DURATION_S = 0.1
EXTRA_TIME_AFTER_FALL_S = 2.0

BASE_SCHEDULE = [(3.0, 150.0), (9.0, -150.0), (15.0, 145.0), (21.0, -145.0)]
TOTAL_TIME_S = 27.0

rng = np.random.RandomState(4)
push_schedule = [(t + rng.uniform(-0.15, 0.15), f + rng.uniform(-8, 8)*np.sign(f)) for t, f in BASE_SCHEDULE]

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

traj.generate_traj(go2, gait, 0.0, V_X_NOM, 0.0, NOMINAL_HEIGHT, 0.0, time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=0.0)
mpc = CentroidalMPC(go2, traj)
U_opt = np.zeros((12, traj.N))

ctrl_i = 0
tau_hold = np.zeros(12)
push_idx = 0
qpos_trace = []
fell_at_ctrl_i = None

max_ctrl_i = int(TOTAL_TIME_S * CTRL_HZ)

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

    if ctrl_i % STEPS_PER_MPC == 0:
        traj.generate_traj(go2, gait, t_now, V_X_NOM, 0.0, NOMINAL_HEIGHT, 0.0, time_step=MPC_DT, slope_deg=0.0, roll_ref_deg=0.0)
        sol = mpc.solve_QP(go2, traj, False)
        N = traj.N
        U_opt = sol["x"].full().flatten()[12*N:].reshape((12,N), order="F")

    f = U_opt[:,0]
    outs = leg_ctrl.compute_all_torques(go2, gait, f, t_now, use_wbc=True, slope_deg=0.0)
    tau = np.zeros(12)
    for leg in LEG_SLICE: tau[LEG_SLICE[leg]] = outs[leg].tau
    tau_hold = np.clip(tau, -TAU_LIM, TAU_LIM)

    for _ in range(CTRL_DECIM):
        mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
        mujoco_go2.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    ctrl_i += 1
    qpos_trace.append(mujoco_go2.data.qpos.copy())

    if d.qpos[2] < 0.15 and fell_at_ctrl_i is None:
        fell_at_ctrl_i = ctrl_i
        print(f"FELL at ctrl_i={ctrl_i}, t={t_now:.3f}s. Continuing for {EXTRA_TIME_AFTER_FALL_S}s more...")

    if fell_at_ctrl_i is not None and ctrl_i >= fell_at_ctrl_i + int(EXTRA_TIME_AFTER_FALL_S * CTRL_HZ):
        print(f"Stopping {EXTRA_TIME_AFTER_FALL_S}s after fall, at ctrl_i={ctrl_i}")
        break

print(f"Completed {ctrl_i} steps ({ctrl_i/CTRL_HZ:.2f}s), {len(qpos_trace)} frames")
qpos_arr = np.array(qpos_trace)
np.savez("/root/go2-convex-mpc/traj_trial4_nominal_extended.npz", qpos=qpos_arr)
print("Saved traj_trial4_nominal_extended.npz")
