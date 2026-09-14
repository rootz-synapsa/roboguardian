import re
from pathlib import Path

# ใช้ so101_new_calib.xml (ไม่มี camera ใน arm แต่เราเพิ่ม overhead_cam ที่ scene แทน)
src_path = Path("vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml")
if not src_path.exists():
    raise SystemExit("SOURCE_MODEL_NOT_FOUND")

xml_text = src_path.read_text()

# FIX: Patch meshdir ให้เป็น relative path จาก models/ ไปยัง vendor/.../assets/
xml_text = re.sub(
    r'meshdir="assets"', 
    'meshdir="../vendor/SO-ARM100/Simulation/SO101/assets"', 
    xml_text
)

# Split XML into logical blocks
pre_wb = xml_text.split('<worldbody>')[0]
wb_content = xml_text.split('<worldbody>')[1].split('</worldbody>')[0]
post_wb = xml_text.split('</worldbody>')[1]

def prefix_names(block, pfx):
    block = re.sub(r'name="([^"]+)"', f'name="{pfx}\\1"', block)
    block = re.sub(r'joint="([^"]+)"', f'joint="{pfx}\\1"', block)
    block = re.sub(r'site="([^"]+)"', f'site="{pfx}\\1"', block)
    block = re.sub(r'body="([^"]+)"', f'body="{pfx}\\1"', block)
    block = re.sub(r'target="([^"]+)"', f'target="{pfx}\\1"', block)
    return block

left_arm = prefix_names(wb_content, 'left_')
right_arm = prefix_names(wb_content, 'right_')

# Handle actuators
act_match = re.search(r'<actuator>(.*?)</actuator>', post_wb, re.DOTALL)
if act_match:
    act_content = act_match.group(1)
    left_act = prefix_names(act_content, 'left_')
    right_act = prefix_names(act_content, 'right_')
    new_act = f"<actuator>\n{left_act}\n{right_act}\n</actuator>"
    post_wb = re.sub(r'<actuator>.*?</actuator>', new_act, post_wb, flags=re.DOTALL)

new_xml = f"""{pre_wb}
<worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="table" pos="0 0 0.4">
      <geom name="table_top" type="box" size="0.6 0.4 0.02" rgba="0.6 0.4 0.2 1" contype="1" conaffinity="1" friction="1 0.1 0.01"/>
    </body>

    <body name="left_arm_base" pos="-0.3 0 0.42">
      {left_arm}
    </body>

    <body name="right_arm_base" pos="0.3 0 0.42">
      {right_arm}
    </body>

    <body name="target_object" pos="0 0 0.45">
      <joint name="target_joint" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <geom name="plate"
            type="cylinder"
            size="0.08 0.01"
            rgba="1 1 1 1"
            mass="0.5"
            contype="0"
            conaffinity="0"/>
    </body>

    <body name="plate" pos="0 0 0.43">
      <joint name="plate_joint" type="free"/>
      <geom name="plate_geom"
            type="cylinder"
            size="0.08 0.01"
            rgba="1 1 1 1"
            mass="0.2"
            contype="0"
            conaffinity="0"
            friction="1 0.1 0.01"/>
    </body>

    <!-- G2 predefined placement target -->
    <body name="place_target" pos="-0.20 0 0.43">
      <geom name="place_target_geom"
            type="cylinder"
            size="0.10 0.002"
            rgba="0 1 0 0.30"
            contype="0"
            conaffinity="0"/>
    </body>

    <camera name="overhead_cam" pos="0 -1.0 1.5" xyaxes="1 0 0 0 0.5 0.866"/>
</worldbody>
{post_wb}
"""

Path("models/dual_so101.xml").write_text(new_xml)
print("✓ models/dual_so101.xml generated successfully (meshdir patched)")
