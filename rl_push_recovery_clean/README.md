# RL Push-Recovery: working code

Validated code for the RL Push-Recovery result described in the main repo README. This folder contains only the final, working scripts. Everything else from the investigation (diagnostics, dead ends, intermediate versions) stayed local and isn't included here.

## Setup

This code depends on 3 files in `src/convex_mpc/` that differ from the base `go2-convex-mpc` repo: `com_trajectory.py` and `gait.py` are modified, and `inplace_terrain_v3.py` is a new file. Copies are included in `convex_mpc_modified/`. Before running anything here, copy those into your `src/convex_mpc/` directory, overwriting the base repo's versions of the first two:

```bash
cp convex_mpc_modified/*.py /path/to/go2-convex-mpc/src/convex_mpc/
```

What changed in each, briefly:
- `com_trajectory.py`: added a `roll_ref_deg` parameter to `generate_traj()`, used to command a nonzero reference roll (needed for the push-recovery residual's roll correction).
- `gait.py`: made `phase_offset` a per-instance attribute instead of a shared global, so each environment instance can have independent gait phase.
- `inplace_terrain_v3.py`: new file, provides `get_flat_model()` for a clean flat-ground scene used by all push-recovery training and evaluation.

## Files

- `mpc_push_recovery_env.py`: the single-push gym environment used to train and validate the base recovery policy (Result 1 in the README).
- `mpc_push_recovery_v1_final.zip`: the trained PPO policy.
- `final_1d_characterization.py`: generalization, inference-latency, and speed-robustness checks on the single-push policy.
- `stage1_record_1d_comparison.py` / `stage2_render_1d_comparison.py`: records and renders the nominal vs RL single-push comparison video.
- `push_recovery_multi_state.py`: the final three-state supervisor (NOMINAL -> RL_RECOVERY -> RETURN_NOMINAL -> NOMINAL) that extends the single-push policy to handle repeated pushes in one episode. Reproduces both randomized 4-push tables (Result 3 in the README):

```bash
  python3 push_recovery_multi_state.py --spacing wide   # 15s between pushes
  python3 push_recovery_multi_state.py --spacing tight  # 6s between pushes
```

- `record_trial4_nominal_extended.py` / `stage2_render_trial4_comparison.py` / `stage2_render_trial4_extended.py`: records and renders the nominal-vs-RL demo videos for the specific tight-spacing trial where nominal falls and RL survives.

## Requirements

Same environment as the base `go2-convex-mpc` repo (see main README), plus `stable-baselines3`.
