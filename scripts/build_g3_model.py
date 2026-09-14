"""
Build G3-specific MuJoCo model.

G3 requires the plate to physically rest on the table after being
disturbed (unlike G2 which uses kinematic carry). Therefore:
  - table_top collision: ENABLED
  - plate_geom collision: ENABLED
  - plate <-> gripper contacts: EXCLUDED (avoid penetration)
  - gripper <-> table contacts: EXCLUDED (avoid residual contacts)
  - target_object collision: DISABLED (legacy)

Reads the ORIGINAL models/dual_so101.xml and produces
models/dual_so101_g3.xml.
"""

from pathlib import Path
import re
import sys


def main():
    src = Path("models/dual_so101.xml")
    dst = Path("models/dual_so101_g3.xml")

    s = src.read_text()

    # 1. Re-enable plate_geom collision (contype=1, conaffinity=1)
    #    The current dual_so101.xml may have contype="0" conaffinity="0"
    #    from our earlier G2 patch. Restore to 1/1.
    s = re.sub(
        r'(<geom\s+name="plate_geom"[^>]*?)contype="0"',
        r'\1contype="1"',
        s,
    )
    s = re.sub(
        r'(<geom\s+name="plate_geom"[^>]*?)conaffinity="0"',
        r'\1conaffinity="1"',
        s,
    )

    # 2. Ensure table_top collision is enabled
    #    (it should already be 1/1 in the original, but be safe)
    s = re.sub(
        r'(<geom\s+name="table_top"[^>]*?)contype="0"',
        r'\1contype="1"',
        s,
    )
    s = re.sub(
        r'(<geom\s+name="table_top"[^>]*?)conaffinity="0"',
        r'\1conaffinity="1"',
        s,
    )

    # 3. Ensure target_object collision stays disabled
    s = re.sub(
        r'(<geom\s+name="plate"[^>]*?)contype="1"',
        r'\1contype="0"',
        s,
    )
    s = re.sub(
        r'(<geom\s+name="plate"[^>]*?)conaffinity="1"',
        r'\1conaffinity="0"',
        s,
    )

    # 4. Add <contact><exclude> section before </mujoco>
    #    This excludes problematic contact pairs while keeping
    #    table<->plate collision active.
    contact_excludes = """
  <contact>
    <!-- G3: plate must rest on table, but should NOT collide with gripper -->
    <exclude body1="plate" body2="left_gripper"/>
    <exclude body1="plate" body2="left_moving_jaw_so101_v1"/>
    <!-- G3: exclude residual gripper<->table contacts -->
    <exclude body1="left_gripper" body2="table"/>
  </contact>
"""

    # Insert before </mujoco>
    if "<contact>" not in s:
        s = s.replace("</mujoco>", contact_excludes + "</mujoco>")
    else:
        # If <contact> already exists, add excludes inside it
        s = re.sub(
            r"(<contact>\s*\n)",
            r"\1" + contact_excludes.strip() + "\n",
            s,
            count=1,
        )

    dst.write_text(s)
    print(f"G3 model written to: {dst.resolve()}")


if __name__ == "__main__":
    main()
