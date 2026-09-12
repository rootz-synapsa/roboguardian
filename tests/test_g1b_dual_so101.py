#!/usr/bin/env python3
"""G1b: dual SO-101 actuator response + disturbance detection (actuator-command based)"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mujoco


def main():
    model = mujoco.MjModel.from_xml_path("models/dual_so101.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pan")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pan")
    left_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pan")
    right_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pan")
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    target_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")

    if -1 in (left_act, right_act, left_jnt, right_jnt, target_body, target_jnt):
        print("G1B2_ACTUATORS=FAIL: missing IDs")
        return 1

    lj_adr = model.jnt_qposadr[left_jnt]
    rj_adr = model.jnt_qposadr[right_jnt]
    tj_adr = model.jnt_qposadr[target_jnt]

    left_before = float(data.qpos[lj_adr])
    right_before = float(data.qpos[rj_adr])

    # Real actuator commands (position servos), NOT qpos edits
    data.ctrl[left_act] = 0.2
    data.ctrl[right_act] = -0.2
    for _ in range(100):
        mujoco.mj_step(model, data)

    left_delta = abs(float(data.qpos[lj_adr]) - left_before)
    right_delta = abs(float(data.qpos[rj_adr]) - right_before)
    left_ok = left_delta > 0.05
    right_ok = right_delta > 0.05
    print(f"left_shoulder_pan  delta: {left_delta:.4f} rad -> {left_ok}")
    print(f"right_shoulder_pan delta: {right_delta:.4f} rad -> {right_ok}")

    if left_ok and right_ok:
        print("G1B2_ACTUATORS=PASS")
    else:
        print("G1B2_ACTUATORS=FAIL")

    # Target pose readable + programmatic disturbance
    pose_before = data.xpos[target_body].copy()
    data.qpos[tj_adr] += 0.10
    mujoco.mj_forward(model, data)
    pose_after = data.xpos[target_body].copy()
    disp = float(np.linalg.norm(pose_after - pose_before))
    dist_ok = disp > 0.05
    print(f"target displacement: {disp:.4f} m -> {dist_ok}")
    if dist_ok:
        print("G1B2_DISTURBANCE=PASS")
    else:
        print("G1B2_DISTURBANCE=FAIL")

    cam_ok = model.ncam >= 1

    overall = left_ok and right_ok and dist_ok and cam_ok
    evidence = {
        "gate": "G1b",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dual_arm_loaded": True,
        "left_actuator_verified": bool(left_ok),
        "right_actuator_verified": bool(right_ok),
        "left_delta_rad": left_delta,
        "right_delta_rad": right_delta,
        "camera_count": int(model.ncam),
        "target_object_readable": True,
        "disturbance_detected": bool(dist_ok),
        "displacement_meters": disp,
        "status": "PASS" if overall else "FAIL",
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/g1b_dual_so101.json", "w") as f:
        json.dump(evidence, f, indent=2)

    print("G1B_FULL=PASS" if overall else "G1B_FULL=FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
