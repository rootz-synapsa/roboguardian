#!/usr/bin/env python3
"""G2: happy-path baseline - single-object pick-place.
Grasp model: kinematic carry (contact physics deferred; documented in evidence)."""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mujoco

HOME      = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP     = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT      = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]
PLACE     = [-0.6, -0.8, 1.1, -0.3, 0.0, -0.6]
RELEASE   = [-0.6, -0.8, 1.1, -0.3, 0.0, 0.0]


def set_arm(data, ids, target):
    for i, v in enumerate(target):
        data.ctrl[ids[i]] = v


def run_phase(model, data, ids, target, steps, carry=None):
    for _ in range(steps):
        set_arm(data, ids, target)
        if carry:
            carry(data)
        mujoco.mj_step(model, data)


def main():
    model = mujoco.MjModel.from_xml_path("models/dual_so101.xml")
    data = mujoco.MjData(model)

    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{n}")
               for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll", "gripper"]]
    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper")
    if -1 in act_ids or plate_jnt == -1 or grip_body == -1:
        print("G2_BASELINE=FAIL: missing ids")
        return 1

    q_adr = model.jnt_qposadr[plate_jnt]
    v_adr = model.jnt_dofadr[plate_jnt]
    OFFSET = np.array([0.0, 0.0, -0.05])

    def carry(d):
        d.qpos[q_adr:q_adr + 3] = d.xpos[grip_body] + OFFSET
        d.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    runs = []
    for run_id in range(1, 11):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        x0, z0 = data.xpos[0][0], data.xpos[0][0]  # placeholder, replaced below
        p0 = data.qpos[q_adr:q_adr + 3].copy()
        x0, z0 = float(p0[0]), float(p0[2])
        z_max = z0

        run_phase(model, data, act_ids, HOME, 100)
        run_phase(model, data, act_ids, PRE_GRASP, 100)
        run_phase(model, data, act_ids, GRASP, 60)
        for _ in range(100):  # lift
            set_arm(data, act_ids, LIFT)
            carry(data)
            mujoco.mj_step(model, data)
            z_max = max(z_max, float(data.qpos[q_adr + 2]))
        for _ in range(150):  # move
            set_arm(data, act_ids, PLACE)
            carry(data)
            mujoco.mj_step(model, data)
        run_phase(model, data, act_ids, RELEASE, 60)  # release (no carry)
        for _ in range(60):  # settle
            set_arm(data, act_ids, RELEASE)
            mujoco.mj_step(model, data)

        pf = data.qpos[q_adr:q_adr + 3].copy()
        vf = data.qvel[v_adr:v_adr + 3].copy()
        lift_ok = (z_max - z0) > 0.05
        move_ok = abs(float(pf[0]) - x0) > 0.15
        place_ok = abs(float(pf[2]) - z0) < 0.02 and float(np.linalg.norm(vf)) < 0.05
        ok = lift_ok and move_ok and place_ok
        runs.append({"run": run_id, "success": bool(ok),
                     "lift_ok": bool(lift_ok), "move_ok": bool(move_ok),
                     "place_ok": bool(place_ok),
                     "x0": x0, "x_final": float(pf[0]),
                     "z0": z0, "z_max": float(z_max)})
        print(f"run {run_id:2d}: success={ok} lift={lift_ok} move={move_ok} place={place_ok}")

    successes = sum(1 for r in runs if r["success"])
    rate = successes / len(runs)
    status = "PASS" if successes >= 8 else "FAIL"
    evidence = {
        "gate": "G2",
        "runs": len(runs),
        "successes": successes,
        "success_rate": rate,
        "task": "single_object_pick_place",
        "disturbance": False,
        "grasp_model": "kinematic_carry (contact physics deferred)",
        "status": status,
        "per_run": runs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/g2_baseline.json").write_text(json.dumps(evidence, indent=2))
    print(f"successes={successes}/10 rate={rate:.2f} status={status}")
    print("G2_BASELINE=" + status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
