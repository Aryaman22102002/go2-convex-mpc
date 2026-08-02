"""
train_mpc_residual.py

Trains the residual correction policy on top of the existing MPC+WBC
pipeline. Much smaller problem than full joint-level RL: 2D action space,
17D observation, and a working baseline controller underneath at every
step (action=0 reproduces unmodified MPC+WBC behavior exactly).
"""
import argparse
import numpy as np
from collections import defaultdict, deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from convex_mpc.mpc_residual_env import MPCResidualEnv


def make_env():
    def _init():
        return MPCResidualEnv()
    return _init


class ResidualEpisodeOutcomeCallback(BaseCallback):
    """Aggregates and prints per-terrain episode outcomes, saturation
    fractions, and MPC/WBC infeasibility rates -- per external review's
    logging requirements. A policy that improves survival but frequently
    drives the QP to infeasibility is not treated as a successful result
    here; both are reported separately."""

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
            print(f"\n[residual episode outcomes @ {self.num_timesteps:,} steps]")
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
                mean_pitch_sat = np.mean([e["frac_pitch_saturated"] for e in buf])
                mean_mpc_infeas = np.mean([e["frac_mpc_infeasible"] for e in buf])
                mean_wbc_infeas = np.mean([e["frac_wbc_infeasible"] for e in buf])
                mean_pitch_accel_err = np.mean([e["mean_pitch_accel_err"] for e in buf])

                label = "ALL" if terrain == "__all__" else terrain
                print(f"  {label:10s} n={n:4d}  success_rate={success_rate:.1%}  "
                      f"mean_len={mean_len:6.1f}  mean_rew={mean_rew:7.2f}")
                print(f"             termination: {reason_str}")
                print(f"             saturation: pitch={mean_pitch_sat:.1%}  "
                      f"| infeasible: mpc={mean_mpc_infeas:.2%}  wbc={mean_wbc_infeas:.2%}  "
                      f"| mean_pitch_accel_err={mean_pitch_accel_err:.3f}")

                self.logger.record(f"residual/{label}_success_rate", success_rate)
                self.logger.record(f"residual/{label}_mean_len", mean_len)
                self.logger.record(f"residual/{label}_pitch_saturation", mean_pitch_sat)
                self.logger.record(f"residual/{label}_mpc_infeasible_frac", mean_mpc_infeas)
                self.logger.record(f"residual/{label}_wbc_infeasible_frac", mean_wbc_infeas)
                self.logger.record(f"residual/{label}_pitch_accel_err", mean_pitch_accel_err)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out", type=str, default="mpc_residual_policy")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    vec_env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)

    if args.resume:
        print(f"Resuming from {args.resume}...")
        model = PPO.load(args.resume, env=vec_env, device="auto")
        model.set_env(vec_env)
    else:
        print("Starting fresh training (pitch-only, 1D action space)...")
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
            # Small network -- this is a 2D action / 17D observation
            # problem, not full joint-level control, so a much smaller
            # policy than the full-locomotion RL work is appropriate.
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1,
            device="auto",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="mpc_residual",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.checkpoint_dir}/best/",
        log_path="training_log/eval/",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    outcome_cb = ResidualEpisodeOutcomeCallback(log_freq=20_000)

    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_cb, eval_cb, outcome_cb],
        progress_bar=True,
    )

    model.save(args.out)
    print(f"Saved to {args.out}.zip")
