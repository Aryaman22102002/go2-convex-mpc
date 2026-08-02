"""
export_pure_rl_step_video.py

Exports a video of the pure RL joint-space policy (go2_terrain_distilled_v3.zip)
on step terrain, forcing step_up (or step_down) specifically rather than
relying on the env's random curriculum sampling -- same rendering pattern
proven in export_stepping_stones_video.py (osmesa, mujoco.Renderer, tracking
camera).
"""
import sys
import numpy as np
import mujoco as mj
from stable_baselines3 import PPO

sys.path.insert(0, "examples")
from go2_terrain_env import Go2TerrainEnv

direction = sys.argv[1] if len(sys.argv) > 1 else "step_up"
step_height = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
policy_path = sys.argv[3] if len(sys.argv) > 3 else "examples/go2_terrain_distilled_v3.zip"
out_name = sys.argv[4] if len(sys.argv) > 4 else f"pure_rl_{direction}.mp4"

RENDER_HZ = 30.0

env = Go2TerrainEnv()
model = PPO.load(policy_path, device="cpu")

# Force step-only terrain by monkey-patching the curriculum, rather than
# manually swapping env.model/env.data after reset() -- reset() has its own
# terrain-specific initial-pose setup logic (RSI etc.) that a post-hoc model
# swap would skip entirely, risking a video where the robot starts in a
# pose that doesn't match the terrain at all.
env._get_curriculum = lambda: {"flat": 0.0, "slope": 0.0, "step": 1.0,
                                "slope_max": 15.0, "step_max": step_height + 0.001}

# reset()'s own logic picks step_up vs step_down randomly and samples a
# height in [0.03, step_max] -- try a few seeds until we get the specific
# direction requested (step_max pinned tightly above so height stays close
# to the requested value regardless of which seed lands).
obs = None
for seed in range(50):
    obs, _ = env.reset(seed=seed)
    if direction in env._terrain_type:
        print(f"Got {env._terrain_type} on seed={seed}")
        break
else:
    print(f"WARNING: couldn't get '{direction}' in 50 seeds, using last reset "
          f"({env._terrain_type}) instead")

print(f"Running pure RL policy on {direction} (height={step_height}m)...")
q_log, t_log = [], []
next_render_t = 0.0
RENDER_DT = 1.0 / RENDER_HZ
done = False
step_count = 0

while not done and step_count < env.max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step_count += 1

    t_now = step_count * env.dt_ctrl
    if t_now + 1e-9 >= next_render_t:
        q_log.append(env.data.qpos.copy())
        t_log.append(t_now)
        next_render_t += RENDER_DT

print(f"Ran {step_count} steps ({'fell' if done and step_count < env.max_steps else 'completed'}), "
      f"rendering {len(q_log)} frames...")

renderer = mj.Renderer(env.model, height=720, width=1280)
data_replay = mj.MjData(env.model)
base_id = env.model.body("base_link").id
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = base_id
cam.distance = 1.5
cam.elevation = -15
cam.azimuth = 90

frames = []
for q in q_log:
    data_replay.qpos[:] = q
    mj.mj_forward(env.model, data_replay)
    renderer.update_scene(data_replay, camera=cam)
    frames.append(renderer.render())

import imageio
imageio.mimsave(out_name, frames, fps=int(RENDER_HZ))
print(f"Saved video to {out_name}")
renderer.close()
