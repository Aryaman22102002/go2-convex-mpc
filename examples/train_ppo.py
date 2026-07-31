"""
train_ppo.py
Train PPO policy for terrain-adaptive locomotion on Go2 quadruped.

Usage (Colab):
    !python train_ppo.py --steps 1000000 --n_envs 2 --xml /path/to/scene.xml \
        --out /content/drive/MyDrive/go2_checkpoints/go2_terrain_policy \
        --checkpoint_dir /content/drive/MyDrive/go2_checkpoints/

Resume after disconnect:
    !python train_ppo.py --steps 1000000 --n_envs 2 --xml /path/to/scene.xml \
        --resume /content/drive/MyDrive/go2_checkpoints/go2_terrain_policy \
        --out /content/drive/MyDrive/go2_checkpoints/go2_terrain_policy \
        --checkpoint_dir /content/drive/MyDrive/go2_checkpoints/
"""

import argparse
import os
import numpy as np
from pathlib import Path


def make_env(xml_path, rank, total_steps_ref=0, steps_per_tick=1):
    def _init():
        from go2_terrain_env import Go2TerrainEnv
        env = Go2TerrainEnv(xml_path=xml_path,
                             total_steps_ref=total_steps_ref,
                             steps_per_tick=steps_per_tick)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",          type=int,   default=1_000_000)
    parser.add_argument("--n_envs",         type=int,   default=2)
    parser.add_argument("--xml",            type=str,   default=None)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--out",            type=str,   default="go2_terrain_policy")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to existing policy zip (without .zip) to resume from")
    parser.add_argument("--curriculum_start", type=int, default=None,
                        help="Override the curriculum's starting step count (e.g. 0 to force "
                             "a fresh flat->easy->hard ramp this session, regardless of the "
                             "resumed model's true cumulative step count). If not set, uses "
                             "the resumed model's real num_timesteps.")
    parser.add_argument("--lr_start", type=float, default=3e-4,
                        help="Learning rate at the start of THIS training call.")
    parser.add_argument("--lr_end", type=float, default=3e-5,
                        help="Learning rate at the end of THIS training call (linear decay). "
                             "Lower than SB3's PPO default; grounded in a published SB3+PPO "
                             "legged-locomotion setup (Go1/Aliengo, lr=5e-5) rather than guessed.")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor. Lowered from the previous 0.99 default, matching "
                             "the same reference setup, for a shorter effective planning horizon.")
    parser.add_argument("--ent_coef_start", type=float, default=0.01)
    parser.add_argument("--ent_coef_end", type=float, default=0.001,
                        help="Entropy coefficient decays linearly over THIS training call so "
                             "exploration tapers off once a good policy is found, instead of "
                             "continuing to churn a policy that already converged.")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="Reset the Adam optimizer state (moment estimates) on resume, "
                             "discarding it from the loaded checkpoint. For the learning-dynamics "
                             "control experiment: a resumed optimizer with very small accumulated "
                             "moments can itself cause near-zero effective updates even with an "
                             "active learning rate, per external review.")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
    from stable_baselines3.common.utils import set_random_seed, get_linear_fn, get_schedule_fn

    class EntropyDecayCallback(BaseCallback):
        """Linearly decays model.ent_coef over the course of THIS training
        call. SB3's PPO does not support a schedule for ent_coef natively
        (only learning_rate/clip_range do) -- this fills that gap so
        exploration tapers off once the policy has found a good solution,
        instead of continuing to push updates that can knock it off a good
        local optimum (the pattern observed in v17: reward peaked partway
        through an 8M-step run, then degraded for the remainder)."""
        def __init__(self, start, end, total_steps, verbose=0):
            super().__init__(verbose)
            self.start = start
            self.end = end
            self.total_steps = total_steps
            self._start_timesteps = None

        def _on_training_start(self):
            self._start_timesteps = self.num_timesteps

        def _on_step(self):
            steps_this_call = self.num_timesteps - self._start_timesteps
            progress = min(1.0, steps_this_call / max(1, self.total_steps))
            self.model.ent_coef = self.start + (self.end - self.start) * progress
            return True

    class TerrainExposureCallback(BaseCallback):
        """Periodically queries actual environment-step exposure per terrain
        category across all parallel workers (not just episode counts, which
        can be misleading if terrain types have very different typical
        episode lengths -- e.g. a broken slope reset failing in 6 steps vs.
        a flat episode running the full 500, meaning episode counts could
        look balanced while real training signal was almost entirely from
        one terrain type). Logs both to console and tensorboard."""
        def __init__(self, log_freq=50_000, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq
            self._last_logged = 0

        def _on_step(self):
            if self.num_timesteps - self._last_logged >= self.log_freq:
                self._last_logged = self.num_timesteps
                try:
                    per_env_counts = self.training_env.env_method("get_terrain_step_counts")
                    totals = {"flat": 0, "slope": 0, "step_up": 0, "step_down": 0}
                    for counts in per_env_counts:
                        for k, v in counts.items():
                            totals[k] += v
                    grand_total = sum(totals.values())
                    if grand_total > 0:
                        print(f"[terrain exposure @ {self.num_timesteps:,} steps] " +
                              ", ".join(f"{k}={v:,} ({100*v/grand_total:.1f}%)"
                                        for k, v in totals.items()))
                        for k, v in totals.items():
                            self.logger.record(f"terrain_exposure/{k}_steps", v)
                            self.logger.record(f"terrain_exposure/{k}_pct", 100*v/grand_total)
                except Exception as e:
                    print(f"[terrain exposure logging failed: {e}]")
            return True

    class VelMatchMonitorCallback(BaseCallback):
        """Periodically queries hip/thigh/calf velocity-error class
        contribution and winning-delta distribution across all parallel
        workers. Per external review: r_vel_match was newly activated after
        being effectively dead throughout the project (measured ~0 reward
        for 95.9% of steps during successful flat walking under the old
        uncalibrated coefficient). This monitors whether training suppresses
        useful hip stabilization motion (watch sum_hip relative to
        sum_thigh/sum_calf) or develops a persistent one-sided phase offset
        (watch delta_counts -- a healthy run should stay roughly balanced
        between -1/+1, not drift toward one side dominating)."""
        def __init__(self, log_freq=50_000, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq
            self._last_logged = 0

        def _on_step(self):
            if self.num_timesteps - self._last_logged >= self.log_freq:
                self._last_logged = self.num_timesteps
                try:
                    per_env_stats = self.training_env.env_method("get_vel_match_stats")
                    total_hip = sum(s["sum_hip"] for s in per_env_stats)
                    total_thigh = sum(s["sum_thigh"] for s in per_env_stats)
                    total_calf = sum(s["sum_calf"] for s in per_env_stats)
                    total_r = sum(s["sum_r_vel_match"] for s in per_env_stats)
                    total_count = sum(s["count"] for s in per_env_stats)
                    delta_totals = {-1: 0, 0: 0, 1: 0}
                    for s in per_env_stats:
                        for d, c in s["delta_counts"].items():
                            delta_totals[d] += c
                    delta_grand_total = sum(delta_totals.values())
                    if total_count > 0:
                        mean_hip = total_hip / total_count
                        mean_thigh = total_thigh / total_count
                        mean_calf = total_calf / total_count
                        mean_r = total_r / total_count
                        print(f"[vel_match @ {self.num_timesteps:,} steps] "
                              f"mean_r_vel_match={mean_r:.4f}  "
                              f"E_hip={mean_hip:.3f}  E_thigh={mean_thigh:.3f}  E_calf={mean_calf:.3f}")
                        self.logger.record("vel_match/mean_r_vel_match", mean_r)
                        self.logger.record("vel_match/E_hip", mean_hip)
                        self.logger.record("vel_match/E_thigh", mean_thigh)
                        self.logger.record("vel_match/E_calf", mean_calf)
                        if delta_grand_total > 0:
                            pct = {d: 100*c/delta_grand_total for d, c in delta_totals.items()}
                            print(f"  winning delta distribution: "
                                  f"-1={pct[-1]:.1f}%  0={pct[0]:.1f}%  +1={pct[1]:.1f}%")
                            for d in (-1, 0, 1):
                                self.logger.record(f"vel_match/delta_{d}_pct", pct[d])
                except Exception as e:
                    print(f"[vel_match logging failed: {e}]")
            return True

    class EpisodeOutcomeCallback(BaseCallback):
        """Periodically queries comprehensive per-terrain-category episode
        outcome stats (survival rate, termination reason breakdown, mean
        episode length/forward distance/yaw drift) and per-foot swing-height
        + diagonal-sync stats, across all parallel workers. Logs everything
        to console and tensorboard so future questions about training
        behavior can be answered directly from these logs, rather than
        needing a bespoke diagnostic script plus another training run."""
        def __init__(self, log_freq=100_000, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq
            self._last_logged = 0

        def _on_step(self):
            if self.num_timesteps - self._last_logged >= self.log_freq:
                self._last_logged = self.num_timesteps
                try:
                    self._log_episode_outcomes()
                    self._log_foot_stats()
                except Exception as e:
                    print(f"[episode outcome logging failed: {e}]")
            return True

        def _log_episode_outcomes(self):
            per_env = self.training_env.env_method("get_episode_outcome_stats")
            cats = ["flat", "slope", "step_up", "step_down"]
            totals = {c: {"episodes": 0, "survived": 0,
                          "terminated_low_height": 0, "terminated_extreme_roll": 0,
                          "terminated_extreme_pitch": 0, "terminated_other": 0,
                          "sum_episode_length": 0, "sum_forward_distance": 0.0,
                          "sum_abs_yaw_drift": 0.0} for c in cats}
            for env_stats in per_env:
                for c in cats:
                    for k, v in env_stats[c].items():
                        totals[c][k] += v

            print(f"\n[episode outcomes @ {self.num_timesteps:,} steps]")
            for c in cats:
                t = totals[c]
                n = t["episodes"]
                if n == 0:
                    continue
                survival_pct = 100 * t["survived"] / n
                mean_len = t["sum_episode_length"] / n
                mean_dist = t["sum_forward_distance"] / n
                mean_yaw = np.degrees(t["sum_abs_yaw_drift"] / n)
                print(f"  {c:10s} n={n:5d}  survival={survival_pct:5.1f}%  "
                      f"mean_len={mean_len:6.1f}  mean_dist={mean_dist:+6.3f}m  "
                      f"mean_|yaw_drift|={mean_yaw:5.1f}deg")
                print(f"             termination breakdown: "
                      f"low_height={100*t['terminated_low_height']/n:.1f}%  "
                      f"extreme_roll={100*t['terminated_extreme_roll']/n:.1f}%  "
                      f"extreme_pitch={100*t['terminated_extreme_pitch']/n:.1f}%  "
                      f"other={100*t['terminated_other']/n:.1f}%")
                self.logger.record(f"episode_outcome/{c}_survival_pct", survival_pct)
                self.logger.record(f"episode_outcome/{c}_mean_length", mean_len)
                self.logger.record(f"episode_outcome/{c}_mean_forward_dist", mean_dist)
                self.logger.record(f"episode_outcome/{c}_mean_abs_yaw_drift_deg", mean_yaw)
                for reason in ("terminated_low_height", "terminated_extreme_roll",
                               "terminated_extreme_pitch", "terminated_other"):
                    self.logger.record(f"episode_outcome/{c}_{reason}_pct", 100*t[reason]/n)

        def _log_foot_stats(self):
            per_env = self.training_env.env_method("get_foot_stats")
            cats = ["flat", "slope", "step_up", "step_down"]
            feet = ["FL", "FR", "RL", "RR"]
            totals = {c: {"sum_height": {f: 0.0 for f in feet},
                          "count_swing": {f: 0 for f in feet},
                          "sum_sync_match": 0.0, "count_sync": 0} for c in cats}
            for env_stats in per_env:
                for c in cats:
                    for f in feet:
                        totals[c]["sum_height"][f] += env_stats[c]["sum_height"][f]
                        totals[c]["count_swing"][f] += env_stats[c]["count_swing"][f]
                    totals[c]["sum_sync_match"] += env_stats[c]["sum_sync_match"]
                    totals[c]["count_sync"] += env_stats[c]["count_sync"]

            for c in cats:
                t = totals[c]
                if t["count_sync"] == 0:
                    continue
                mean_heights = {f: (t["sum_height"][f] / t["count_swing"][f]
                                     if t["count_swing"][f] > 0 else 0.0) for f in feet}
                front_mean = (mean_heights["FL"] + mean_heights["FR"]) / 2
                rear_mean  = (mean_heights["RL"] + mean_heights["RR"]) / 2
                ratio_pct = (100 * (rear_mean / front_mean - 1)) if front_mean > 1e-6 else 0.0
                mean_sync = 100 * t["sum_sync_match"] / t["count_sync"]
                print(f"  {c:10s} swing heights FL={mean_heights['FL']:.3f} "
                      f"FR={mean_heights['FR']:.3f} RL={mean_heights['RL']:.3f} "
                      f"RR={mean_heights['RR']:.3f}  rear/front={ratio_pct:+.1f}%  "
                      f"sync_match={mean_sync:.1f}%")
                self.logger.record(f"foot_stats/{c}_rear_front_ratio_pct", ratio_pct)
                self.logger.record(f"foot_stats/{c}_sync_match_pct", mean_sync)
                for f in feet:
                    self.logger.record(f"foot_stats/{c}_{f}_mean_swing_height", mean_heights[f])

    class LearningDynamicsCallback(BaseCallback):
        """Diagnoses whether PPO is actually updating the policy at all, per
        external review. Near-zero approx_kl + zero clip_fraction + frozen
        eval metrics over millions of steps is ambiguous on its own -- it
        could mean genuine convergence, or it could mean training has
        silently stalled (tiny effective LR, resumed optimizer state with
        vanishingly small Adam updates, etc). The parameter-update norm ratio
        ||theta_{t+1} - theta_t|| / ||theta_t|| is the most decisive single
        metric: if this is ~0, the actor genuinely is not changing,
        regardless of what the loss curves show. Also logs per-joint-class
        (hip/thigh/calf) action std directly from the policy's log_std, since
        decayed entropy coefficient doesn't necessarily mean zero real
        exploration if the learned std remains substantial."""
        def __init__(self, log_freq=50_000, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq
            self._last_logged = 0
            self._prev_params = None

        def _flat_params(self):
            import torch
            with torch.no_grad():
                return torch.cat([p.detach().flatten().cpu()
                                   for p in self.model.policy.parameters()])

        def _on_step(self):
            if self.num_timesteps - self._last_logged >= self.log_freq:
                self._last_logged = self.num_timesteps
                try:
                    import torch
                    current_params = self._flat_params()
                    if self._prev_params is not None:
                        delta_norm = torch.norm(current_params - self._prev_params).item()
                        theta_norm = torch.norm(self._prev_params).item()
                        ratio = delta_norm / max(theta_norm, 1e-12)
                        print(f"[learning_dynamics @ {self.num_timesteps:,} steps] "
                              f"||delta_theta||/||theta||={ratio:.3e}  "
                              f"(over last {self.log_freq:,} steps)")
                        self.logger.record("learning_dynamics/param_update_ratio", ratio)
                        if ratio < 1e-5:
                            print("  !! WARNING: parameter update ratio is essentially zero -- "
                                  "the actor is not meaningfully changing, regardless of loss curves")
                    self._prev_params = current_params

                    # Per-joint-class action std, actuator order [hip,thigh,calf] x 4 legs
                    log_std = self.model.policy.log_std.detach().cpu().numpy()
                    std = np.exp(log_std)
                    mean_hip   = float(np.mean(std[0::3]))
                    mean_thigh = float(np.mean(std[1::3]))
                    mean_calf  = float(np.mean(std[2::3]))
                    print(f"  action std: hip={mean_hip:.4f}  thigh={mean_thigh:.4f}  "
                          f"calf={mean_calf:.4f}  overall_mean={std.mean():.4f}")
                    self.logger.record("learning_dynamics/action_std_hip", mean_hip)
                    self.logger.record("learning_dynamics/action_std_thigh", mean_thigh)
                    self.logger.record("learning_dynamics/action_std_calf", mean_calf)
                    self.logger.record("learning_dynamics/action_std_overall", float(std.mean()))
                except Exception as e:
                    print(f"[learning_dynamics logging failed: {e}]")
            return True

    set_random_seed(42)

    # ------------------------------------------------------------------
    # Determine the model's TRUE cumulative step count before creating any
    # environments, so curriculum progression (gated on total_steps inside
    # Go2TerrainEnv) reflects real training history instead of silently
    # restarting from 0 on every --resume, which is what happened previously
    # (make_env never passed total_steps_ref at all).
    # ------------------------------------------------------------------
    resume_path = args.resume + ".zip" if args.resume else None
    starting_steps = 0
    if args.curriculum_start is not None:
        starting_steps = args.curriculum_start
        print(f"Curriculum start explicitly overridden to {starting_steps:,} "
              f"(ignoring resumed model's true step count, if any).")
    elif resume_path and os.path.exists(resume_path):
        print(f"Peeking at resumed model's step count from {resume_path}...")
        _peek_model = PPO.load(args.resume, device="cpu")
        starting_steps = int(_peek_model.num_timesteps)
        print(f"  Resumed model has {starting_steps:,} true cumulative steps -- "
              f"curriculum will start from this point, not 0.")
        del _peek_model

    print(f"Creating {args.n_envs} parallel environments "
          f"(curriculum starting_steps={starting_steps:,}, steps_per_tick={args.n_envs})...")
    env_fns = [make_env(args.xml, i, total_steps_ref=starting_steps, steps_per_tick=args.n_envs)
               for i in range(args.n_envs)]
    vec_env  = VecMonitor(SubprocVecEnv(env_fns))
    eval_env = VecMonitor(SubprocVecEnv([make_env(args.xml, 99,
                                                   total_steps_ref=starting_steps,
                                                   steps_per_tick=args.n_envs)]))

    lr_schedule_fn = get_linear_fn(args.lr_start, args.lr_end, 1.0)

    # Resume from checkpoint or create new model
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        model = PPO.load(args.resume, env=vec_env, device="auto")
        model.set_env(vec_env)

        # PPO.load() restores the CHECKPOINT'S OWN saved hyperparameters
        # (including its original constant learning_rate and gamma=0.99) --
        # overriding them here directly rather than relying on the loaded
        # values, per diagnosed peak-then-degrade pattern in v17.
        model.lr_schedule = get_schedule_fn(lr_schedule_fn)
        model.gamma = args.gamma
        model.ent_coef = args.ent_coef_start

        if args.reset_optimizer:
            print("  --reset_optimizer set: reinitializing Adam optimizer state "
                  "(discarding loaded moment estimates) for the learning-dynamics control test.")
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(),
                lr=args.lr_start,
                **model.policy.optimizer_kwargs,
            )

        print(f"  Loaded. Overriding schedule: lr {args.lr_start:.1e}->{args.lr_end:.1e}, "
              f"gamma={args.gamma}, ent_coef {args.ent_coef_start}->{args.ent_coef_end}. "
              f"Continuing training for {args.steps:,} more steps.")
    else:
        print("Starting fresh training...")
        model = PPO(
            policy        = "MlpPolicy",
            env           = vec_env,
            learning_rate = lr_schedule_fn,
            n_steps       = 2048,
            batch_size    = 512,
            n_epochs      = 10,
            gamma         = args.gamma,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = args.ent_coef_start,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            policy_kwargs = dict(
                net_arch      = [dict(pi=[256, 256], vf=[256, 256])],
                activation_fn = __import__("torch").nn.ELU,
            ),
            tensorboard_log = "training_log/",
            verbose         = 1,
            device          = "auto",
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model parameters: {n_params}")
    print(f"Training for {args.steps:,} steps across {args.n_envs} envs...")

    # Save checkpoints to Drive so they survive Colab disconnects
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(50_000 // args.n_envs, 1),
        save_path   = args.checkpoint_dir,
        name_prefix = "go2_ppo",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(args.checkpoint_dir, "best/"),
        log_path             = "training_log/eval/",
        eval_freq            = max(100_000 // args.n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )
    terrain_exposure_cb = TerrainExposureCallback(log_freq=50_000)
    vel_match_monitor_cb = VelMatchMonitorCallback(log_freq=50_000)
    episode_outcome_cb = EpisodeOutcomeCallback(log_freq=100_000)
    learning_dynamics_cb = LearningDynamicsCallback(log_freq=50_000)

    entropy_cb = EntropyDecayCallback(
        start=args.ent_coef_start,
        end=args.ent_coef_end,
        total_steps=args.steps,
    )

    model.learn(
        total_timesteps = args.steps,
        callback        = [checkpoint_cb, eval_cb, entropy_cb, terrain_exposure_cb,
                           vel_match_monitor_cb, episode_outcome_cb, learning_dynamics_cb],
        progress_bar    = True,
        reset_num_timesteps = (args.resume is None),  # don't reset counter on resume
    )

    model.save(args.out)
    print(f"\nTraining complete. Policy saved to {args.out}.zip")

if __name__ == "__main__":
    main()
