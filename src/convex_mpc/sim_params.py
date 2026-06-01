"""
sim_params.py
Central configuration for all simulation experiments.
Change MU here to switch surface conditions -- everything else updates automatically.
"""

import re
from pathlib import Path

# --------------------------------------------------------------------------------
# Surface friction
# Typical values: carpet/rubber ~0.8, tile ~0.5, wet floor ~0.3, ice ~0.1
# MU_SAFE is set slightly below MU to keep the QP solution strictly inside
# the friction cone boundary (avoids floating point violations at the edge).
# --------------------------------------------------------------------------------
MU      = 0.8
MU_SAFE = 0.78

# --------------------------------------------------------------------------------
# Gait velocities
# Sideways motion demands full lateral friction budget, so we back off at low mu.
# Forward and rotation use smaller effective lateral forces so 0.8 m/s stays safe.
# --------------------------------------------------------------------------------
VEL_FORWARD  = 0.8
VEL_SIDEWAY  = 0.4 if MU >= 0.5 else 0.35   # 0.35 m/s is the stable limit at mu=0.3
VEL_ROTATION = 4.0                            # rad/s

# --------------------------------------------------------------------------------
# Output folder tag -- keeps mu=0.8 and mu=0.3 results in separate folders
# --------------------------------------------------------------------------------
def get_friction_tag():
    return f"mu{int(MU*10):02d}"


# --------------------------------------------------------------------------------
# XML patcher
# MuJoCo reads friction from two places: the foot geom in go2.xml and the
# floor geom in scene.xml. Both need to match MU or the contact friction
# (geometric mean of the two) won't equal what the WBC assumes.
# --------------------------------------------------------------------------------
REPO      = Path(__file__).resolve().parents[2]
GO2_XML   = REPO / "models" / "MJCF" / "go2" / "go2.xml"
SCENE_XML = REPO / "models" / "MJCF" / "go2" / "scene.xml"

def apply_friction_to_xml():
    # go2.xml -- patch default geom friction and foot geom friction
    go2_text = GO2_XML.read_text()
    go2_text = re.sub(r'friction="[\d.]+"(\s+margin)',
                      f'friction="{MU:.1f}"\\1', go2_text)
    go2_text = re.sub(r'friction="[\d.]+ 0\.02 0\.01"',
                      f'friction="{MU:.1f} 0.02 0.01"', go2_text)
    GO2_XML.write_text(go2_text)

    # scene.xml -- patch floor geom friction
    scene_text = SCENE_XML.read_text()
    if 'friction=' in scene_text:
        scene_text = re.sub(r'friction="[\d.]+ 0\.005 0\.0001"',
                            f'friction="{MU:.1f} 0.005 0.0001"', scene_text)
    else:
        scene_text = scene_text.replace(
            'type="plane"',
            f'type="plane" friction="{MU:.1f} 0.005 0.0001"'
        )
    SCENE_XML.write_text(scene_text)

    print(f"[sim_params] XML friction set to {MU:.1f}")
