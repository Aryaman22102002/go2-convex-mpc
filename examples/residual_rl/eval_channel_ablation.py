"""
eval_channel_ablation.py

Per external review: before widening the height-correction bound, determine
whether height saturation is actually CAUSAL (the policy genuinely needs
more height authority) or merely COMPENSATORY/a learned boundary bias. This
uses the SAME trained checkpoint under four conditions -- no retraining --
masking one action channel at evaluation time, not at training time.

Runs on identical slope seeds (both uphill and downhill, tracked
separately), recording the specific diagnostics requested:
  - survival, episode length
  - low-height / excessive-pitch termination rates
  - MPC / WBC infeasibility rates
  - applied height saturation
  - mean SIGNED height correction and mean SIGNED pitch correction
    (sign matters: reviewer wants to check if height saturates with the
    correct recovery sign, and whether it precedes or follows failure)
"""
import argparse
import numpy as np
from stable_baselines3 import PPO

from convex_mpc.mpc_residual_env import MPCResidualEnv

CONDITIONS = ["both", "height_only", "pitch_only", "zero_residual"]


def run_episode(env, policy, condition, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_steps = 0
    height_corrs, pitch_corrs = [], []
    height_saturated_steps = 0
    n_mpc_infeasible, n_wbc_infeasible = 0, 0
    term_reason = None

    while True:
        if condition == "zero_residual":
            action = np.zeros(2, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action, dtype=np.float32)
            if condition == "height_only":
                action[0] = 0.0   # zero the pitch channel at evaluation time
            elif condition == "pitch_only":
                action[1] = 0.0   # zero the height channel at evaluation time

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1

        height_corrs.append(info["applied_height_corr_m"])
        pitch_corrs.append(info["applied_pitch_corr_deg"])
        if abs(action[1]) > 0.95:
            height_saturated_steps += 1
        if not info["mpc_feasible_this_step"]:
            n_mpc_infeasible += 1
        if not info["wbc_feasible_this_step"]:
            n_wbc_infeasible += 1

        if terminated or truncated:
            term_reason = info["residual_episode"]["term_reason"]
            break

    return {
        "steps": n_steps,
        "term_reason": term_reason,
        "reward": total_reward,
        "mean_height_corr": np.mean(height_corrs) if height_corrs else 0.0,
        "mean_pitch_corr": np.mean(pitch_corrs) if pitch_corrs else 0.0,
        "frac_height_saturated": height_saturated_steps / max(n_steps, 1),
        "frac_mpc_infeasible": n_mpc_infeasible / max(n_steps, 1),
        "frac_wbc_infeasible": n_wbc_infeasible / max(n_steps, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    model = PPO.load(args.policy, device="cpu")
    curriculum = {"flat": 0.0, "slope": 1.0, "step": 0.0}

    results = {cond: {"uphill": [], "downhill": []} for cond in CONDITIONS}

    for trial in range(args.n_trials):
        seed = 2000 + trial  # SAME seed across all 4 conditions -> identical slope draw
        for condition in CONDITIONS:
            env = MPCResidualEnv(terrain_curriculum=curriculum)
            r = run_episode(env, model, condition, seed=seed)
            # Read the TRUE slope sign directly from the env (reliable),
            # not inferred from the policy's own correction sign
            direction = "uphill" if env._true_slope_deg > 0 else "downhill"
            results[condition][direction].append(r)
            print(f"Trial {trial} [{condition:14s}] dir={direction:8s} "
                  f"steps={r['steps']:4d} reason={r['term_reason']:16s} "
                  f"mean_h_corr={r['mean_height_corr']:+.4f} mean_p_corr={r['mean_pitch_corr']:+.2f} "
                  f"h_sat={r['frac_height_saturated']:.1%}")

    print("\n=== Channel Ablation Summary (slope, identical seeds across conditions) ===")
    for condition in CONDITIONS:
        for direction in ["uphill", "downhill"]:
            trials = results[condition][direction]
            if not trials:
                continue
            n = len(trials)
            success = np.mean([t["term_reason"] == "success_timeout" for t in trials])
            mean_steps = np.mean([t["steps"] for t in trials])
            low_height = np.mean([t["term_reason"] == "low_height" for t in trials])
            extreme_pitch = np.mean([t["term_reason"] == "extreme_pitch" for t in trials])
            mpc_infeas = np.mean([t["frac_mpc_infeasible"] for t in trials])
            wbc_infeas = np.mean([t["frac_wbc_infeasible"] for t in trials])
            h_sat = np.mean([t["frac_height_saturated"] for t in trials])
            mean_h = np.mean([t["mean_height_corr"] for t in trials])
            mean_p = np.mean([t["mean_pitch_corr"] for t in trials])

            print(f"\n{condition:14s} | {direction:8s} (n={n})")
            print(f"  success_rate={success:.1%}  mean_len={mean_steps:.1f}  "
                  f"low_height_term={low_height:.1%}  extreme_pitch_term={extreme_pitch:.1%}")
            print(f"  mpc_infeasible={mpc_infeas:.2%}  wbc_infeasible={wbc_infeas:.2%}  "
                  f"height_saturation={h_sat:.1%}")
            print(f"  mean_signed_height_corr={mean_h:+.4f}m  mean_signed_pitch_corr={mean_p:+.2f}deg")
