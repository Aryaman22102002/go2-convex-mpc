"""
train_ppo.py
Train PPO policy for terrain-adaptive locomotion on Go2 quadruped.

Designed to run on Colab T4 GPU.
Uses Stable-Baselines3 with SubprocVecEnv for parallel training.

Usage (Colab):
    !python train_ppo.py --steps 5000000 --n_envs 8 --xml /path/to/scene.xml

Outputs:
    checkpoints/go2_ppo_<steps>.zip   -- periodic checkpoints
    go2_terrain_policy.zip            -- final policy
    training_log/                     -- tensorboard logs
"""

import argparse
import os
import numpy as np
from pathlib import Path

def make_env(xml_path, rank, total_steps_ref):
    def _init():
        from go2_terrain_env import Go2TerrainEnv
        env = Go2TerrainEnv(xml_path=xml_path, total_steps_ref=total_steps_ref)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int,   default=5_000_000)
    parser.add_argument("--n_envs",  type=int,   default=8)
    parser.add_argument("--xml",     type=str,   default=None)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--out",     type=str,   default="go2_terrain_policy")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback
    )
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(42)

    # Shared step counter for curriculum
    # SB3 SubprocVecEnv runs in separate processes so we use a simple
    # approximation: each env tracks its own steps, curriculum advances
    # based on total_timesteps passed to learn()
    total_steps_ref = [0]   # mutable reference

    print(f"Creating {args.n_envs} parallel environments...")
    env_fns = [make_env(args.xml, i, 0) for i in range(args.n_envs)]

    from stable_baselines3.common.vec_env import SubprocVecEnv
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Single eval env
    from go2_terrain_env import Go2TerrainEnv
    eval_env = VecMonitor(SubprocVecEnv([make_env(args.xml, 99, 0)]))

    # PPO hyperparameters tuned for locomotion
    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = args.lr,
        n_steps         = 2048,        # steps per env before update
        batch_size      = 512,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,        # encourage exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(
            net_arch    = [dict(pi=[256, 256], vf=[256, 256])],
            activation_fn = __import__("torch").nn.ELU,
        ),
        tensorboard_log = "training_log/",
        verbose         = 1,
        device          = "auto",      # uses GPU if available
    )

    print(f"Model parameters: {sum(p.numel() for p in model.policy.parameters())}")
    print(f"Training for {args.steps:,} steps across {args.n_envs} envs...")
    print(f"Curriculum: flat -> slope+-5 -> slope+-10+step -> slope+-15+step")

    # Callbacks
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq   = 500_000 // args.n_envs,
        save_path   = "checkpoints/",
        name_prefix = "go2_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = "checkpoints/best/",
        log_path             = "training_log/eval/",
        eval_freq            = 100_000 // args.n_envs,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(
        total_timesteps     = args.steps,
        callback            = [checkpoint_cb, eval_cb],
        progress_bar        = True,
    )

    model.save(args.out)
    print(f"\nTraining complete. Policy saved to {args.out}.zip")
    print("To use on your laptop, download the .zip file.")

if __name__ == "__main__":
    main()
