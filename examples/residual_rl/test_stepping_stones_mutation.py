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

print("\n=== Verifying layout correctness (with runway) ===")
N_RUNWAY = 3
START_X = -0.5
set_stepping_stones(model, stone_length=0.3, gap_width=0.15, stone_width=0.6,
                     n_runway_stones=N_RUNWAY, start_x=START_X)
expected_x = START_X
for i, sid in enumerate(stone_ids):
    pos = model.geom_pos[sid]
    size = model.geom_size[sid]
    expected_center_x = expected_x + 0.15  # stone_length/2
    assert np.isclose(pos[0], expected_center_x, atol=1e-6), \
        f"stone {i}: expected x={expected_center_x}, got {pos[0]}"
    assert np.isclose(size[0], 0.15, atol=1e-6)
    assert np.isclose(size[1], 0.3, atol=1e-6)
    this_gap = 0.0 if i < N_RUNWAY - 1 else 0.15
    expected_x += 0.3 + this_gap
print(f"Layout correct: {len(stone_ids)} stones, first {N_RUNWAY} form a solid runway, "
      f"gap_width=0.15 starts after that")

# Confirm the runway is genuinely continuous (no gap within it)
for i in range(N_RUNWAY - 1):
    s_end = model.geom_pos[stone_ids[i]][0] + model.geom_size[stone_ids[i]][0]
    s_next_start = model.geom_pos[stone_ids[i+1]][0] - model.geom_size[stone_ids[i+1]][0]
    assert np.isclose(s_end, s_next_start, atol=1e-6), \
        f"runway stone {i}->​{i+1} should have zero gap, got {s_next_start - s_end}"
print(f"Confirmed: stones 0-{N_RUNWAY-1} form a continuous, gap-free runway")

# Confirm the first REAL gap (after the runway) matches gap_width
real_gap_idx = N_RUNWAY - 1
s_end = model.geom_pos[stone_ids[real_gap_idx]][0] + model.geom_size[stone_ids[real_gap_idx]][0]
s_next_start = model.geom_pos[stone_ids[real_gap_idx+1]][0] - model.geom_size[stone_ids[real_gap_idx+1]][0]
print(f"First real gap (after runway) spans x=[{s_end:.3f}, {s_next_start:.3f}], "
      f"width={s_next_start-s_end:.4f}")

print("\n=== Testing varied gap widths (measuring first REAL gap, after runway) ===")
for gap_width in [0.05, 0.15, 0.25, 0.35]:
    set_stepping_stones(model, stone_length=0.3, gap_width=gap_width, stone_width=0.6,
                         n_runway_stones=N_RUNWAY)
    idx = N_RUNWAY - 1
    s_end = model.geom_pos[stone_ids[idx]][0] + model.geom_size[stone_ids[idx]][0]
    s_next_start = model.geom_pos[stone_ids[idx+1]][0] - model.geom_size[stone_ids[idx+1]][0]
    actual_gap = s_next_start - s_end
    assert np.isclose(actual_gap, gap_width, atol=1e-6), f"gap mismatch: {actual_gap} vs {gap_width}"
    print(f"gap_width={gap_width}: OK (measured gap={actual_gap:.4f})")

print("\n=== Testing lateral jitter (should not affect runway stones) ===")
rng = np.random.default_rng(0)
set_stepping_stones(model, stone_length=0.3, gap_width=0.15, stone_width=0.6,
                     lateral_jitter=0.1, rng=rng, n_runway_stones=N_RUNWAY)
y_positions = [model.geom_pos[sid][1] for sid in stone_ids]
print(f"Stone y-positions with jitter: {np.round(y_positions, 3)}")
assert all(abs(y_positions[i]) < 1e-9 for i in range(N_RUNWAY)), "runway stones should have zero jitter"
assert any(abs(y) > 1e-6 for y in y_positions[N_RUNWAY:]), "post-runway jitter should produce nonzero y offsets"
print("Lateral jitter working (runway unaffected, post-runway stones jittered)")

print("\n=== Stress test: 300 in-place layout switches, one process ===")
rng = np.random.default_rng(1)
for i in range(300):
    gap_width = rng.uniform(0.05, 0.35)
    stone_length = rng.uniform(0.2, 0.4)
    set_stepping_stones(model, stone_length=stone_length, gap_width=gap_width,
                         stone_width=0.6, lateral_jitter=0.05, rng=rng, n_runway_stones=N_RUNWAY)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

print("Completed 300 in-place stepping-stone layout switches with no crash.")
