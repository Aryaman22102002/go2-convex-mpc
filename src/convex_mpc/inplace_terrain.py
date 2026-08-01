"""
inplace_terrain.py

In-place terrain mutation for the universal terrain base model. Loads a
single MuJoCo model ONCE and mutates its floor/platform geometry per
episode via the MuJoCo Python API, instead of loading a different XML
file per terrain. Fixes a resource-exhaustion bug found to be triggered
by the total count of distinct mj.MjModel.from_xml_path() calls within a
process -- validated via test_inplace_terrain_mutation.py (300 in-place
switches, one process, no crash; geometrically verified correct).

Also enables genuinely CONTINUOUS terrain randomization (not a fixed
discrete library), since mutation parameters can be sampled fresh every
episode with no disk I/O cost.
"""
import numpy as np
import mujoco as mj

UNIVERSAL_XML_PATH = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_universal_terrain.xml"

_UNIVERSAL_MODEL = None
_FLOOR_ID = None
_PLATFORM_ID = None


def get_universal_model():
    """Loads the universal terrain model once per process, caches it."""
    global _UNIVERSAL_MODEL, _FLOOR_ID, _PLATFORM_ID
    if _UNIVERSAL_MODEL is None:
        _UNIVERSAL_MODEL = mj.MjModel.from_xml_path(UNIVERSAL_XML_PATH)
        _FLOOR_ID = mj.mj_name2id(_UNIVERSAL_MODEL, mj.mjtObj.mjOBJ_GEOM, "floor")
        _PLATFORM_ID = mj.mj_name2id(_UNIVERSAL_MODEL, mj.mjtObj.mjOBJ_GEOM, "platform")
    return _UNIVERSAL_MODEL


def set_flat(model):
    """Floor level, platform parked out of the way, non-colliding."""
    model.geom_quat[_FLOOR_ID] = [1, 0, 0, 0]
    model.geom_pos[_PLATFORM_ID] = [1000, 1000, -1000]
    model.geom_contype[_PLATFORM_ID] = 0
    model.geom_conaffinity[_PLATFORM_ID] = 0


def set_slope(model, slope_deg):
    """Tilt the floor about the Y axis by slope_deg, platform parked away."""
    slope_rad = np.radians(slope_deg)
    half = slope_rad / 2.0
    model.geom_quat[_FLOOR_ID] = [np.cos(half), 0, np.sin(half), 0]
    model.geom_pos[_PLATFORM_ID] = [1000, 1000, -1000]
    model.geom_contype[_PLATFORM_ID] = 0
    model.geom_conaffinity[_PLATFORM_ID] = 0


def set_step(model, step_height, direction="up"):
    """Floor level, platform moved into position and re-enabled to form a step."""
    model.geom_quat[_FLOOR_ID] = [1, 0, 0, 0]
    if direction == "up":
        z = step_height / 2.0
        model.geom_pos[_PLATFORM_ID] = [2.0, 0.0, z]
        model.geom_size[_PLATFORM_ID] = [2.0, 2.0, step_height / 2.0]
    else:
        z = -step_height / 2.0
        model.geom_pos[_PLATFORM_ID] = [-1.0, 0.0, z]
        model.geom_size[_PLATFORM_ID] = [3.0, 2.0, step_height / 2.0]
    model.geom_contype[_PLATFORM_ID] = 1
    model.geom_conaffinity[_PLATFORM_ID] = 1
