import os
import numpy as np
import mujoco as mj

LIBRARY_DIR = os.path.join("models", "MJCF", "go2", "terrain_library")

_MODEL_CACHE = {}


def _load_mjb(fname):
    if fname in _MODEL_CACHE:
        return _MODEL_CACHE[fname]
    path = os.path.join(LIBRARY_DIR, fname)
    model = mj.MjModel.from_binary_path(path)
    _MODEL_CACHE[fname] = model
    return model


def get_flat_model():
    return _load_mjb("flat.mjb")


def get_slope_model(slope_deg):
    angle = int(round(np.clip(slope_deg, -15, 15)))
    model = _load_mjb(f"slope_{angle}.mjb")
    return model, float(angle)


def get_step_model(step_height, direction="up"):
    available_cm = [2, 3, 4, 5, 6, 7, 8]
    height_cm = step_height * 100.0
    nearest_cm = min(available_cm, key=lambda c: abs(c - height_cm))
    model = _load_mjb(f"step_{direction}_{nearest_cm}cm.mjb")
    return model, nearest_cm / 100.0
