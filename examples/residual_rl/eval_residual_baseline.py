"""
eval_residual_baseline.py

Compares, on IDENTICAL terrain conditions (same seed per trial):
  1. Nominal controller with zero residual (action forced to [0,0])
  2. Trained deterministic residual policy

Per external review: the meaningful metric is the INCREMENTAL improvement
the residual produces, not absolute performance alone.
"""
import argparse
import numpy as np
from stable_baselines3 import PPO

from convex_mpc.mpc_residual_env import MPCResidualEnv


def run_episode(env, policy=None, seed=None):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_steps = 0
    term_reason = None
    while True:
        if policy is None:
            action = np.zeros(2, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            term_reason = info["residual_episode"]["term_reason"]
            break
    return {"reward": total_reward, "steps": n_steps, "term_reason": term_reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--terrain", type=str, default="slope",
                         choices=["flat", "slope", "step_up", "step_down"])
    args = parser.parse_args()

    model = PPO.load(args.policy, device="cpu")

    if args.terrain in ("step_up", "step_down"):
        curriculum = {"flat": 0.0, "slope": 0.0, "step": 1.0}
    else:
        curriculum = {"flat": 1.0 if args.terrain == "flat" else 0.0,
                       "slope": 1.0 if args.terrain == "slope" else 0.0,
                       "step": 0.0}

    env_zero = MPCResidualEnv(terrain_curriculum=curriculum)
    env_learned = MPCResidualEnv(terrain_curriculum=curriculum)

    zero_results, learned_results = [], []
    for trial in range(args.n_trials):
        seed = 1000 + trial  # SAME seed for both -> identical terrain draw per trial
        r_zero = run_episode(env_zero, policy=None, seed=seed)
        r_learned = run_episode(env_learned, policy=model, seed=seed)
        zero_results.append(r_zero)
        learned_results.append(r_learned)
        print(f"Trial {trial}: zero_residual steps={r_zero['steps']:4d} "
              f"reason={r_zero['term_reason']:16s} | "
              f"learned steps={r_learned['steps']:4d} reason={r_learned['term_reason']}")

    def summarize(results, label):
        success = np.mean([r["term_reason"] == "success_timeout" for r in results])
        mean_steps = np.mean([r["steps"] for r in results])
        mean_reward = np.mean([r["reward"] for r in results])
        print(f"\n{label}: success_rate={success:.1%}  mean_steps={mean_steps:.1f}  mean_reward={mean_reward:.2f}")
        return success

    print(f"\n=== Terrain: {args.terrain} ({args.n_trials} identical-seed trials) ===")
    s_zero = summarize(zero_results, "Zero residual (nominal MPC+WBC)")
    s_learned = summarize(learned_results, "Learned residual")
    print(f"\nIncremental improvement: {(s_learned - s_zero)*100:+.1f} percentage points survival")
