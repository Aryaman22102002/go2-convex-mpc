"""
eval_pitch_only_uphill_downhill.py

Zero-residual (nominal MPC+WBC) vs the trained pitch-only policy, on
identical seeded slope terrain, split explicitly by uphill/downhill --
the direct comparison this training run was built to answer.
"""
import sys
import numpy as np
from stable_baselines3 import PPO

from convex_mpc.mpc_residual_env import MPCResidualEnv


def run_episode(env, policy, use_policy, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_steps = 0
    term_reason = None
    while True:
        if use_policy:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.zeros(1, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            term_reason = info["residual_episode"]["term_reason"]
            break
    return {"steps": n_steps, "term_reason": term_reason, "reward": total_reward}


if __name__ == "__main__":
    policy_path = sys.argv[1] if len(sys.argv) > 1 else "mpc_residual_pitch_only.zip"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    model = PPO.load(policy_path, device="cpu")
    curriculum = {"flat": 0.0, "slope": 1.0, "step": 0.0}

    results = {"uphill": {"zero": [], "trained": []}, "downhill": {"zero": [], "trained": []}}

    for trial in range(n_trials):
        seed = 5000 + trial
        for use_policy, label in [(False, "zero"), (True, "trained")]:
            env = MPCResidualEnv(terrain_curriculum=curriculum)
            r = run_episode(env, model, use_policy, seed=seed)
            direction = "uphill" if env._true_slope_deg > 0 else "downhill"
            results[direction][label].append(r)
            print(f"Trial {trial} [{label:8s}] dir={direction:8s} steps={r['steps']:4d} "
                  f"reason={r['term_reason']}")

    print("\n=== Zero-residual vs Trained (pitch-only), split by slope direction ===")
    for direction in ["uphill", "downhill"]:
        for label in ["zero", "trained"]:
            trials = results[direction][label]
            if not trials:
                continue
            n = len(trials)
            success = np.mean([t["term_reason"] == "success_timeout" for t in trials])
            mean_steps = np.mean([t["steps"] for t in trials])
            print(f"{direction:8s} | {label:8s} (n={n}): success_rate={success:.1%}  mean_steps={mean_steps:.1f}")
        print()
