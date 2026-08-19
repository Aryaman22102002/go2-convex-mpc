import sys
sys.path.insert(0, 'src')
import numpy as np
from stable_baselines3 import PPO
from convex_mpc.mpc_push_recovery_env import MPCPushRecoveryEnv

policy = PPO.load("mpc_push_recovery_v1_final.zip", device="cpu")

WX_TH, VY_TH, ROLL_TH = 56.85, 0.188, 2.08
PUSH_FORCE = 150.0

def run_and_save(name, use_policy, seed=0):
    env = MPCPushRecoveryEnv()
    env._sample_push = lambda: PUSH_FORCE
    obs, _ = env.reset(seed=seed)

    qpos_trace = []
    gate_open, gate_open_until = False, -1.0
    for i in range(1200):
        d = env.mujoco_go2.data
        qw, qx, qy, qz = d.qpos[3:7]
        roll_now = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)))
        wx_now = np.degrees(env.go2.current_config.base_ang_vel[0])
        vy_now = env.go2.current_config.base_vel[1]
        t_now = float(env.mujoco_go2.data.time)
        if not gate_open and (abs(wx_now) > WX_TH or abs(vy_now) > VY_TH or abs(roll_now) > ROLL_TH):
            gate_open = True
            gate_open_until = t_now + 1.25
        if gate_open and t_now > gate_open_until:
            gate_open = False

        if use_policy and gate_open:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        qpos_trace.append(env.mujoco_go2.data.qpos.copy())
        if terminated or truncated:
            print(f"  {name}: ended, ctrl_i={env._ctrl_i}, reason={info['residual_episode']['term_reason']}")
            break

    qpos_arr = np.array(qpos_trace)
    np.savez(f"/root/go2-convex-mpc/traj_{name}.npz", qpos=qpos_arr)
    print(f"  Saved traj_{name}.npz, {len(qpos_trace)} frames")

print(f"Recording nominal response to {PUSH_FORCE}N push...")
run_and_save("1d_nominal_150N", use_policy=False)

print(f"Recording RL-gated response to {PUSH_FORCE}N push...")
run_and_save("1d_rl_gated_150N", use_policy=True)

print("Stage 1 complete.")
