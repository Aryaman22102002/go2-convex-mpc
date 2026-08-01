"""
Generates slope/step terrain XML variants based on the clean flat scene,
for testing the MPC+WBC controller on the same terrain types used to
evaluate the RL policy.
"""
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

BASE_XML = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene_flat_clean.xml"


def make_terrain_xml(terrain_type, slope_deg=0.0, step_height=0.0):
    tree = ET.parse(BASE_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if terrain_type == "flat":
        pass  # base scene is already flat

    elif terrain_type == "slope":
        floor = worldbody.find("geom")  # the 'floor' geom is first
        floor.set("euler", f"0 {slope_deg} 0")

    elif terrain_type == "step_up":
        box = ET.SubElement(worldbody, "geom")
        box.set("name", "platform")
        box.set("type", "box")
        box.set("pos", f"2.0 0 {step_height/2:.4f}")
        box.set("size", f"2.0 2.0 {step_height/2:.4f}")
        box.set("friction", "0.8 0.005 0.0001")

    elif terrain_type == "step_down":
        floor_low = ET.SubElement(worldbody, "geom")
        floor_low.set("name", "floor_low")
        floor_low.set("type", "plane")
        floor_low.set("pos", f"3.0 0 {-step_height:.4f}")
        floor_low.set("size", "10 10 0.1")
        floor_low.set("friction", "0.8 0.005 0.0001")

        platform = ET.SubElement(worldbody, "geom")
        platform.set("name", "platform")
        platform.set("type", "box")
        platform.set("pos", f"-1.0 0 {-step_height/2:.4f}")
        platform.set("size", f"3.0 2.0 {step_height/2:.4f}")
        platform.set("friction", "0.8 0.005 0.0001")

    fd, path = tempfile.mkstemp(suffix=".xml", dir=str(Path(BASE_XML).parent))
    tree.write(path)
    return path
