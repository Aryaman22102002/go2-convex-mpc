import sys
sys.path.insert(0, 'src')
import numpy as np
import mujoco as mj
import imageio
from convex_mpc.inplace_terrain_v3 import get_flat_model

FPS = 30

def render_from_traj(name, source_hz):
    data_npz = np.load(f"/root/go2-convex-mpc/traj_{name}.npz", allow_pickle=True)
    qpos_arr = data_npz["qpos"]

    subsample_factor = source_hz / FPS
    n_frames_out = int(len(qpos_arr) / subsample_factor)
    indices = (np.arange(n_frames_out) * subsample_factor).astype(int)
    indices = indices[indices < len(qpos_arr)]
    qpos_arr = qpos_arr[indices]

    model = get_flat_model()
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.distance = 2.2
    cam.azimuth = 90
    cam.elevation = -15

    frames = []
    for qpos in qpos_arr:
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        cam.lookat[:] = [qpos[0] + 0.3, 0.0, 0.2]
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        frames.append(pixels)

    renderer.close()
    out_path = f"/root/go2-convex-mpc/video_{name}_realtime.mp4"
    imageio.mimsave(out_path, frames, fps=FPS)
    duration_s = len(frames) / FPS
    print(f"  Saved {out_path}: {len(frames)} frames, {duration_s:.1f}s playback (real-time)")

render_from_traj("trial4_nominal_extended", source_hz=200)
print("Done.")
