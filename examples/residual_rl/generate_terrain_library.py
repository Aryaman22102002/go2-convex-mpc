"""
Pre-generates a fixed library of terrain XML files, run ONCE before
training. Episodes then randomly pick from this permanent set rather than
generating and deleting a fresh temp file every single reset -- the
repeated create/delete churn of many small XML files in the same directory
as the mesh assets appears to be the actual trigger for a recurring MuJoCo
resource-exhaustion bug encountered several times in this project (never
observed when simply reloading an already-existing, never-deleted file
many times).
"""
import numpy as np
from convex_mpc.mpc_terrain_gen import make_terrain_xml

N_SLOPE = 40
N_STEP_UP = 20
N_STEP_DOWN = 20

if __name__ == "__main__":
    # Each entry: (kind, path, true_slope_deg)
    entries = [("flat", "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml", 0.0)]

    for d in np.linspace(-15, 15, N_SLOPE):
        d = float(d)
        if abs(d) < 2.0:
            d = np.sign(d) * 2.0 if d != 0 else 5.0
        path = make_terrain_xml("slope", slope_deg=d)
        entries.append(("slope", path, d))

    for h in np.linspace(0.03, 0.08, N_STEP_UP):
        path = make_terrain_xml("step_up", step_height=float(h))
        entries.append(("step_up", path, 0.0))

    for h in np.linspace(0.03, 0.08, N_STEP_DOWN):
        path = make_terrain_xml("step_down", step_height=float(h))
        entries.append(("step_down", path, 0.0))

    with open("terrain_library.txt", "w") as f:
        for kind, path, slope_deg in entries:
            f.write(f"{kind}\t{path}\t{slope_deg}\n")

    n_slope = sum(1 for e in entries if e[0] == "slope")
    n_up = sum(1 for e in entries if e[0] == "step_up")
    n_down = sum(1 for e in entries if e[0] == "step_down")
    print(f"Generated terrain library: {n_slope} slope, {n_up} step_up, {n_down} step_down files.")
    print("Saved manifest to terrain_library.txt")
