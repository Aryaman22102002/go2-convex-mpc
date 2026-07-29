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

    entropy_cb = EntropyDecayCallback(
        start=args.ent_coef_start,
        end=args.ent_coef_end,
        total_steps=args.steps,
    )

    model.learn(
        total_timesteps = args.steps,
        callback        = [checkpoint_cb, eval_cb, entropy_cb],
        progress_bar    = True,
        reset_num_timesteps = (args.resume is None),  # don't reset counter on resume
    )

    model.save(args.out)
    print(f"\nTraining complete. Policy saved to {args.out}.zip")

if __name__ == "__main__":
    main()
