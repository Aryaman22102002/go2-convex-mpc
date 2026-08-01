"""
build_universal_terrain_xml.py

Run ONCE, offline, to generate a single base XML that supports flat,
slope, AND step terrain entirely through runtime geometry mutation
(model.geom_pos / geom_quat / geom_size / geom_contype via the MuJoCo
Python API), rather than loading a different XML file per terrain.

Adds a permanent "platform" spare geom to the existing clean flat scene,
initially parked far away and non-colliding. At runtime, this geom gets
moved into position and re-enabled for step terrain, or left parked for
flat/slope terrain (where only the floor's tilt needs to change).
"""
import xml.etree.ElementTree as ET

BASE_XML = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml"
OUT_XML = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_universal_terrain.xml"

tree = ET.parse(BASE_XML)
root = tree.getroot()
worldbody = root.find("worldbody")

# Spare platform geom: parked far below/away and non-colliding by default
# (contype=0, conaffinity=0 disables collision without needing to remove
# the geom entirely -- geoms can't be added/removed at runtime, but their
# pos/size/contype CAN be mutated).
platform = ET.SubElement(worldbody, "geom")
platform.set("name", "platform")
platform.set("type", "box")
platform.set("pos", "1000 1000 -1000")   # parked far away
platform.set("size", "1.0 1.0 0.01")
platform.set("friction", "0.8 0.005 0.0001")
platform.set("contype", "0")
platform.set("conaffinity", "0")

tree.write(OUT_XML)
print(f"Wrote universal terrain base to {OUT_XML}")

# Sanity check: confirm it loads and report geom count / the new geom's index
import mujoco as mj
m = mj.MjModel.from_xml_path(OUT_XML)
print(f"Loaded OK, ngeom={m.ngeom}")
platform_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "platform")
floor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
print(f"platform geom id={platform_id}, floor geom id={floor_id}")
