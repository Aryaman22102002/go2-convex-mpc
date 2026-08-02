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

# Keep the floor plane ACTIVE, but make it FINITE and position it to
# cover only the runway/approach region, ending exactly where the box
# stones begin (fix: sinking the plane away entirely forced 100%
# continuous box contact, which direct testing showed causes real WBC
# instability -- "max iterations reached"/"solved inaccurate" repeatedly,
# even on a single giant flat box with no gaps at all. Every prior
# successful use of box geoms in this project, e.g. step terrain, only
# ever used box contact BRIEFLY, backed by a stable plane majority of the
# time. This keeps that same pattern: stable, proven plane contact for
# the approach, box contact only for the actual challenge region.
#
# MuJoCo plane size="sx sy spacing": sx/sy are HALF-extents (0 = infinite
# in that direction). A finite sx lets the plane end at a specific x.
PLANE_END_X = -0.5  # plane covers x in [-100.5, -0.5]; stones begin at x=-0.5
floor = worldbody.find("geom")  # 'floor' geom is first, per existing convention
plane_half_extent = 100.0
plane_center_x = PLANE_END_X - plane_half_extent
floor.set("pos", f"{plane_center_x:.3f} 0 0")
floor.set("size", f"{plane_half_extent} 0 0.05")  # finite in x, infinite in y

for i in range(N_STONES):
    stone = ET.SubElement(worldbody, "geom")
    stone.set("name", f"stone_{i}")
    stone.set("type", "box")
    # Nominal starting layout: evenly spaced along +x, will be
    # repositioned/resized per episode via the MuJoCo API
    stone.set("pos", f"{i * 0.5:.3f} 0 -0.025")
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
