import sys
sys.path.insert(0, 'src')
import time
import numpy as np
from stable_baselines3 import PPO
import convex_mpc.mpc_push_recovery_env as recovery_module
from convex_mpc.mpc_push_recovery_env import MPCPushRecoveryEnv

policy = PPO.load("mpc_push_recovery_v1_final.zip", device="cpu")

WX_TH, VY_TH, ROLL_TH = 56.85, 0.188, 2.08

def run_gated_trial(push_force, use_policy, seed=0):
    env = MPCPushRecoveryEnv()
    env._sample_push = lambda: push_force
    obs, _ = env.reset(seed=seed)
    peak_roll = 0.0
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
        peak_roll = max(peak_roll, abs(roll_now))
        if terminated or truncated:
            success = info["residual_episode"]["term_reason"] == "success_timeout"
            return success, peak_roll
    return True, peak_roll

print("=== 1. Generalization beyond training range (60N, 200N) ===")
for push in [60.0, 200.0]:
    for mode, use_policy in [("nominal", False), ("RL-gated", True)]:
        successes, peak_rolls = [], []
        for trial in range(10):
            success, peak_roll = run_gated_trial(push, use_policy, seed=trial)
            successes.append(success)
            peak_rolls.append(peak_roll)
        print(f"  push={push:5.0f}N {mode:>9}: success_rate={np.mean(successes):.1%}  mean_peak_roll={np.mean(peak_rolls):.2f}deg")

print("\n=== 2. Inference latency vs MPC QP solve time ===")
env = MPCPushRecoveryEnv()
obs, _ = env.reset(seed=0)
n_calls = 200
t0 = time.perf_counter()
for _ in range(n_calls):
    action, _ = policy.predict(obs, deterministic=True)
t1 = time.perf_counter()
policy_us = (t1 - t0) / n_calls * 1e6
print(f"  Policy inference: {policy_us:.1f} microseconds/call")

t0 = time.perf_counter()
sol = env.mpc.solve_QP(env.go2, env.traj, False)
t1 = time.perf_counter()
qp_us = (t1 - t0) * 1e6
print(f"  MPC QP solve (single call): {qp_us:.1f} microseconds")
print(f"  Policy overhead as % of QP solve time: {policy_us/qp_us*100:.2f}%")

print("\n=== 3. Robustness to different nominal walking speeds ===")
for v_nom in [0.4, 0.6, 0.8]:
    recovery_module.V_X_NOM = v_nom
    for mode, use_policy in [("nominal", False), ("RL-gated", True)]:
        successes, peak_rolls = [], []
        for trial in range(10):
            success, peak_roll = run_gated_trial(150.0, use_policy, seed=trial)
            successes.append(success)
            peak_rolls.append(peak_roll)
        print(f"  v_nom={v_nom:.1f} {mode:>9}: success_rate={np.mean(successes):.1%}  mean_peak_roll={np.mean(peak_rolls):.2f}deg")
recovery_module.V_X_NOM = 0.6
