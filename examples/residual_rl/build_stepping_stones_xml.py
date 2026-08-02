"""
build_stepping_stones_xml.py

Run ONCE, offline, to generate a base XML supporting stepping-stone/gap
terrain entirely through runtime geometry mutation. A fixed number of
"stone" geoms are pre-authored in a nominal row; at runtime their
positions, sizes, and the gaps between them are mutated per episode.

This targets a genuinely different terrain type than slope/step: one
where MPC+WBC's fixed swing-foot touchdown heuristic (gait.py's
nominal-hip-offset-plus-drift-correction target) has NO mechanism to
avoid a bad foothold or a gap at all -- footstep placement was never
part of what the MPC/WBC optimization solves for. This is a case where
a learned policy can do something the classical controller structurally
cannot, not just do the same thing better.
"""
import xml.etree.ElementTree as ET

BASE_XML = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml"
OUT_XML = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_stepping_stones.xml"

N_STONES = 10  # fixed count, pre-authored; spacing/size mutated at runtime

tree = ET.parse(BASE_XML)
root = tree.getroot()
worldbody = root.find("worldbody")

# Sink the original flat floor far below and non-colliding -- stepping
# stones become the ONLY walkable surface. A foot that misses a stone
# has nothing to catch it (matching a real gap), rather than silently
# landing on a hidden flat floor underneath.
floor = worldbody.find("geom")  # 'floor' geom is first, per existing convention
floor.set("pos", "0 0 -1000")
floor.set("contype", "0")
floor.set("conaffinity", "0")

for i in range(N_STONES):
    stone = ET.SubElement(worldbody, "geom")
    stone.set("name", f"stone_{i}")
    stone.set("type", "box")
    # Nominal starting layout: evenly spaced along +x, will be
    # repositioned/resized per episode via the MuJoCo API
    stone.set("pos", f"{i * 0.5:.3f} 0 0")
    stone.set("size", "0.15 0.3 0.05")
    stone.set("friction", "0.8 0.005 0.0001")

tree.write(OUT_XML)
print(f"Wrote stepping-stone terrain base to {OUT_XML} with {N_STONES} stones")

# Sanity check: confirm it loads and report geom ids
import mujoco as mj
m = mj.MjModel.from_xml_path(OUT_XML)
print(f"Loaded OK, ngeom={m.ngeom}")
stone_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"stone_{i}") for i in range(N_STONES)]
print(f"Stone geom ids: {stone_ids}")
