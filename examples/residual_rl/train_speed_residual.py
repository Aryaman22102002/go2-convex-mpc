"""
train_speed_residual.py

Trains the speed-residual policy (Delta v_x only) per external review's
design, following the forward-speed sweep's clean confirmation that
commanded speed is the dominant lever for genuine-uphill survival.
"""
import argparse
import numpy as np
import torch
from collections import defaultdict, deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from convex_mpc.mpc_speed_residual_env import MPCSpeedResidualEnv


def make_env(slope_weight=0.8, uphill_frac=0.8):
    remaining = (1.0 - slope_weight) / 2.0
    curriculum = {"flat": remaining, "slope": slope_weight, "step": remaining}
    def _init():
        return MPCSpeedResidualEnv(terrain_curriculum=curriculum, uphill_frac=uphill_frac)
    return _init


class SpeedResidualOutcomeCallback(BaseCallback):
    """Same style of per-terrain outcome/transition-fraction logging as
    the pitch-only training, adapted for speed-specific fields."""

    def __init__(self, log_freq=20_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_logged = 0
        self._buffers = defaultdict(lambda: deque(maxlen=200))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("residual_episode")
            if ep is None:
                continue
            terrain = ep["terrain"]
            self._buffers[terrain].append(ep)
            self._buffers["__all__"].append(ep)

        if self.num_timesteps - self._last_logged >= self.log_freq:
            self._last_logged = self.num_timesteps
            print(f"\n[speed residual outcomes @ {self.num_timesteps:,} steps]")
            real_terrains = ["flat", "slope", "step_up", "step_down"]
            transitions_by_terrain = {t: sum(e["l"] for e in self._buffers[t]) for t in real_terrains}
            total_transitions = sum(transitions_by_terrain.values()) or 1
            print("  [transition fractions] " + "  ".join(
                f"{t}={transitions_by_terrain[t]/total_transitions:.1%}" for t in real_terrains))
            for terrain in ["flat", "slope", "step_up", "step_down", "__all__"]:
                buf = self._buffers[terrain]
                if not buf:
                    continue
                n = len(buf)
                mean_len = np.mean([e["l"] for e in buf])
                mean_rew = np.mean([e["r"] for e in buf])
                success_rate = np.mean([e["term_reason"] == "success_timeout" for e in buf])
                reasons = defaultdict(int)
                for e in buf:
                    reasons[e["term_reason"]] += 1
                reason_str = "  ".join(f"{k}={v/n:.1%}" for k, v in sorted(reasons.items()))
                mean_speed_sat = np.mean([e["frac_speed_saturated"] for e in buf])
                mean_mpc_infeas = np.mean([e["frac_mpc_infeasible"] for e in buf])
                mean_wbc_infeas = np.mean([e["frac_wbc_infeasible"] for e in buf])
                mean_v_x_actual = np.mean([e["final_v_x_actual"] for e in buf])
                mean_v_x_cmd = np.mean([e["final_v_x_cmd"] for e in buf])

                label = "ALL" if terrain == "__all__" else terrain
                print(f"  {label:10s} n={n:4d}  success_rate={success_rate:.1%}  "
                      f"mean_len={mean_len:6.1f}  mean_rew={mean_rew:7.2f}")
                print(f"             termination: {reason_str}")
                print(f"             saturation: speed={mean_speed_sat:.1%}  "
                      f"| infeasible: mpc={mean_mpc_infeas:.2%}  wbc={mean_wbc_infeas:.2%}  "
                      f"| mean_v_x_cmd={mean_v_x_cmd:.3f}  mean_v_x_actual={mean_v_x_actual:.3f}")

                self.logger.record(f"speed_residual/{label}_success_rate", success_rate)
                self.logger.record(f"speed_residual/{label}_mean_len", mean_len)
                self.logger.record(f"speed_residual/{label}_speed_saturation", mean_speed_sat)
                self.logger.record(f"speed_residual/{label}_mpc_infeasible_frac", mean_mpc_infeas)
                self.logger.record(f"speed_residual/{label}_wbc_infeasible_frac", mean_wbc_infeas)
                self.logger.record(f"speed_residual/{label}_mean_v_x_cmd", mean_v_x_cmd)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--resume_weights", type=str, default=None)
    parser.add_argument("--out", type=str, default="mpc_speed_residual_policy")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_speed/")
    parser.add_argument("--slope_weight", type=float, default=0.8,
                         help="Curriculum weight on slope terrain (remainder split evenly flat/step)")
    parser.add_argument("--uphill_frac", type=float, default=0.8,
                         help="Fraction of slope episodes drawn from genuine uphill")
    args = parser.parse_args()

    vec_env = SubprocVecEnv([make_env(args.slope_weight, args.uphill_frac) for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    eval_env = SubprocVecEnv([make_env(args.slope_weight, args.uphill_frac)])
    eval_env = VecMonitor(eval_env)

    if args.resume_weights:
        print(f"Constructing fresh PPO and loading extracted weights from {args.resume_weights}...")
    else:
        print("Starting fresh training (speed-only residual, 1D action space)...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        device="cpu",
    )
    if args.resume_weights:
        state_dict = torch.load(args.resume_weights, map_location="cpu")
        model.policy.load_state_dict(state_dict)
        print("Weights loaded successfully into freshly-constructed PPO.")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="mpc_speed_residual",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.checkpoint_dir}/best/",
        log_path="training_log_speed/eval/",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    outcome_cb = SpeedResidualOutcomeCallback(log_freq=20_000)

    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_cb, eval_cb, outcome_cb],
        progress_bar=True,
    )

    model.save(args.out)
    print(f"Saved to {args.out}.zip")
