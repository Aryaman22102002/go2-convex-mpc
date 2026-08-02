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
STEPPING_STONES_XML_PATH = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_stepping_stones.xml"

_UNIVERSAL_MODEL = None
_FLOOR_ID = None
_PLATFORM_ID = None

_STEPPING_STONES_MODEL = None


def get_stepping_stones_model():
    """Loads the stepping-stones terrain model once per process, cached
    separately from the slope/step universal model (different base XML)."""
    global _STEPPING_STONES_MODEL
    if _STEPPING_STONES_MODEL is None:
        _STEPPING_STONES_MODEL = mj.MjModel.from_xml_path(STEPPING_STONES_XML_PATH)
    return _STEPPING_STONES_MODEL


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


_N_STONES = 10
_STONE_IDS = None


def get_stone_ids(model):
    global _STONE_IDS
    if _STONE_IDS is None:
        _STONE_IDS = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"stone_{i}")
                      for i in range(_N_STONES)]
    return _STONE_IDS


def set_stepping_stones(model, stone_length=0.3, gap_width=0.15, stone_width=0.6,
                         lateral_jitter=0.0, rng=None, n_runway_stones=3, start_x=-0.5):
    """Lays out N_STONES stones in a row along +x, STARTING at start_x
    (fix: confirmed via direct foot-position measurement that the robot's
    rear feet sit at roughly x=-0.19 relative to its base at spawn -- a
    stone sequence starting exactly at x=0 leaves the rear feet standing
    on nothing at all from the very first instant, regardless of gap
    width or stone length, since it's a margin problem behind the robot,
    not a within-terrain gap problem). start_x=-0.5 gives comfortable
    clearance for the robot's full foot spread.

    The first n_runway_stones are placed with ZERO gap between them (a
    continuous solid runway) before any real gaps begin. stone_length is
    the stone's extent along x (direction of travel); stone_width is its
    extent along y. Optional per-stone lateral (y) jitter for added
    difficulty (applied only past the runway), requiring an rng if used.

    Stone top surface is placed at z=0 (matching flat-ground convention
    used everywhere else in this project).
    """
    stone_height = 0.05
    stone_ids = get_stone_ids(model)
    x = start_x
    for i, sid in enumerate(stone_ids):
        y = 0.0
        if i >= n_runway_stones and lateral_jitter > 0.0 and rng is not None:
            y = rng.uniform(-lateral_jitter, lateral_jitter)
        model.geom_pos[sid] = [x + stone_length / 2.0, y, -stone_height / 2.0]
        model.geom_size[sid] = [stone_length / 2.0, stone_width / 2.0, stone_height / 2.0]
        model.geom_contype[sid] = 1
        model.geom_conaffinity[sid] = 1
        this_gap = 0.0 if i < n_runway_stones - 1 else gap_width
        x += stone_length + this_gap
