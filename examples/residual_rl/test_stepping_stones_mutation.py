"""
test_stepping_stones_mutation.py

Validates the stepping-stone terrain in-place mutation before building
the full footstep-placement RL environment around it: correct geometry,
and repeated switching in one process without the resource-exhaustion
bug (same validation pattern as test_inplace_terrain_mutation.py).
"""
import numpy as np
import mujoco as mj

from convex_mpc.inplace_terrain import get_stepping_stones_model, set_stepping_stones, get_stone_ids

model = get_stepping_stones_model()
stone_ids = get_stone_ids(model)
print(f"Loaded stepping-stones model. {len(stone_ids)} stone geom ids: {stone_ids}")

print("\n=== Verifying layout correctness ===")
set_stepping_stones(model, stone_length=0.3, gap_width=0.15, stone_width=0.6)
expected_x = 0.0
for i, sid in enumerate(stone_ids):
    pos = model.geom_pos[sid]
    size = model.geom_size[sid]
    expected_center_x = expected_x + 0.15  # stone_length/2
    assert np.isclose(pos[0], expected_center_x, atol=1e-6), \
        f"stone {i}: expected x={expected_center_x}, got {pos[0]}"
    assert np.isclose(size[0], 0.15, atol=1e-6)
    assert np.isclose(size[1], 0.3, atol=1e-6)
    expected_x += 0.3 + 0.15  # stone_length + gap_width
print(f"Layout correct: {len(stone_ids)} stones, stone_length=0.3, gap_width=0.15")

gap_start = 0.3  # end of first stone
gap_end = 0.3 + 0.15  # start of second stone
print(f"First gap spans x=[{gap_start:.3f}, {gap_end:.3f}] -- a foot landing here should find nothing")

print("\n=== Testing varied gap widths ===")
for gap_width in [0.05, 0.15, 0.25, 0.35]:
    set_stepping_stones(model, stone_length=0.3, gap_width=gap_width, stone_width=0.6)
    s0_end = model.geom_pos[stone_ids[0]][0] + model.geom_size[stone_ids[0]][0]
    s1_start = model.geom_pos[stone_ids[1]][0] - model.geom_size[stone_ids[1]][0]
    actual_gap = s1_start - s0_end
    assert np.isclose(actual_gap, gap_width, atol=1e-6), f"gap mismatch: {actual_gap} vs {gap_width}"
    print(f"gap_width={gap_width}: OK (measured gap={actual_gap:.4f})")

print("\n=== Testing lateral jitter ===")
rng = np.random.default_rng(0)
set_stepping_stones(model, stone_length=0.3, gap_width=0.15, stone_width=0.6,
                     lateral_jitter=0.1, rng=rng)
y_positions = [model.geom_pos[sid][1] for sid in stone_ids]
print(f"Stone y-positions with jitter: {np.round(y_positions, 3)}")
assert any(abs(y) > 1e-6 for y in y_positions), "jitter should produce nonzero y offsets"
print("Lateral jitter working")

print("\n=== Stress test: 300 in-place layout switches, one process ===")
rng = np.random.default_rng(1)
for i in range(300):
    gap_width = rng.uniform(0.05, 0.35)
    stone_length = rng.uniform(0.2, 0.4)
    set_stepping_stones(model, stone_length=stone_length, gap_width=gap_width,
                         stone_width=0.6, lateral_jitter=0.05, rng=rng)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

print("Completed 300 in-place stepping-stone layout switches with no crash.")
