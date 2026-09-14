"""
G3 Metric Inspection — 1 trial only
=====================================

Runs exactly ONE G3 trial with the corrected PLACE values and prints
all diagnostic values needed to determine whether the 109.3223 m
plate_error_3d is caused by:
  (1) simulation explosion (contact solver accumulation), or
  (2) body-id vs joint-id mixup.

Usage:
    python experiments/g3_metric_inspection.py \
        --model models/dual_so101_no_table_collision.xml \
        --trigger before_grasp \
        --delta 0.10 0.0 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]

# PLACE from clean-scene IK (frozen in G2, 10/10 PASS)
PLACE = [
    -0.0006804320,
    -1.3891631148,
    1.3904820914,
    1.5105929073,
    0.0039194352,
    -0.1745329776,
]

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

CARRY_OFFSET = np.array([0.0, 0.0, 0.048])
TRANSPORT_STEPS = 2000

FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}


def build_model(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)

    act_ids, jnt_ids = [], []
    for name in JOINT_NAMES:
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{name}"
        )
        if act_id < 0:
            raise LookupError(f"actuator not found: left_{name}")
        jnt_id = int(model.actuator_trnid[act_id, 0])
        act_ids.append(act_id)
        jnt_ids.append(jnt_id)

    ee_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe"
    )
    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "place_target"
    )
    plate_jnt = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint"
    )
    # IMPORTANT: also get the BODY id for the plate, not just the joint id.
    # data.xpos[...] must be indexed by body id, not joint id.
    plate_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "plate"
    )

    if -1 in (ee_site, target_body, plate_jnt, plate_body):
        raise LookupError(
            f"required model id missing: "
            f"ee_site={ee_site}, target_body={target_body}, "
            f"plate_jnt={plate_jnt}, plate_body={plate_body}"
        )

    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        model.actuator_forcelimited[act_id] = True
        model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, ee_site, target_body, plate_jnt, plate_body


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/dual_so101_no_table_collision.xml",
    )
    parser.add_argument("--trigger", default="before_grasp")
    parser.add_argument(
        "--delta", type=float, nargs=3, default=[0.10, 0.0, 0.0]
    )
    args = parser.parse_args()

    model, act_ids, ee_site, target_body, plate_jnt, plate_body = (
        build_model(args.model)
    )
    data = mujoco.MjData(model)

    q_adr = model.jnt_qposadr[plate_jnt]

    # Print id info so we can spot any mixup
    print("=== ID CHECK ===")
    print(f"plate_joint id (JOINT) : {plate_jnt}")
    print(f"plate body id (BODY)   : {plate_body}")
    print(f"plate jnt_qposadr      : {q_adr}")
    print(f"target_body id (BODY)  : {target_body}")
    print(f"ee_site id (SITE)      : {ee_site}")
    print()

    injector = DisturbanceInjector(
        object_joint_name="plate_joint",
        trigger=args.trigger,
        delta=tuple(args.delta),
        seed=42,
        noise_std=0.0,
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    plate_pose_pre = data.qpos[q_adr:q_adr + 3].copy().tolist()
    print("=== INITIAL STATE ===")
    print(f"plate_pose_pre (qpos) : {plate_pose_pre}")
    print(f"plate xpos (body)     : {data.xpos[plate_body].tolist()}")
    print()

    # Approach
    injector.maybe_fire(model, data, "before_approach")
    for _ in range(100):
        set_arm(data, act_ids, HOME)
        mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP)
        mujoco.mj_step(model, data)

    # Grasp (disturbance fires here)
    injector.maybe_fire(model, data, "before_grasp")
    print("=== AFTER DISTURBANCE ===")
    print(f"plate qpos xyz  : {data.qpos[q_adr:q_adr+3].tolist()}")
    print(f"plate xpos body : {data.xpos[plate_body].tolist()}")
    print(f"disturbance event: {injector.event}")
    print()

    for _ in range(60):
        set_arm(data, act_ids, GRASP)
        mujoco.mj_step(model, data)

    # Lift
    injector.maybe_fire(model, data, "after_grasp")
    for _ in range(100):
        set_arm(data, act_ids, LIFT)
        mujoco.mj_step(model, data)

    # Transport
    injector.maybe_fire(model, data, "before_transport")
    for _ in range(TRANSPORT_STEPS):
        set_arm(data, act_ids, PLACE)
        mujoco.mj_step(model, data)

    # Final state
    injector.maybe_fire(model, data, "before_placement")

    print("=== FINAL STATE ===")
    print(f"target_pos (body)     : {data.xpos[target_body].tolist()}")
    print(f"plate qpos xyz        : {data.qpos[q_adr:q_adr+3].tolist()}")
    print(f"plate xpos (body)     : {data.xpos[plate_body].tolist()}")
    print(f"ee_pos (site)         : {data.site_xpos[ee_site].tolist()}")
    print()

    # ===== DIAGNOSTIC VALUES (per user request) =====
    print("=== DIAGNOSTIC VALUES ===")
    qvel_norm = float(np.linalg.norm(data.qvel))
    print(f"qvel_norm             : {qvel_norm:.6f}")
    print(f"ncon                  : {data.ncon}")

    # max contact force
    max_force = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address >= 0:
            # efc_force is indexed per constraint dimension
            # For a single contact, typically 1-6 constraints
            n_con = c.dim if hasattr(c, 'dim') else 1
            f = data.efc_force[c.efc_address : c.efc_address + n_con]
            fnorm = float(np.linalg.norm(f))
            if fnorm > max_force:
                max_force = fnorm
    print(f"max_contact_force     : {max_force:.4f}")

    nan_in_qpos = bool(np.isnan(data.qpos).any())
    print(f"nan_in_qpos           : {nan_in_qpos}")
    print(f"plate xpos (raw)      : {data.xpos[plate_body].tolist()}")
    print(f"target xpos (raw)     : {data.xpos[target_body].tolist()}")
    print()

    # ===== ERROR CALCULATIONS =====
    target_pos = data.xpos[target_body].copy()
    plate_qpos_pos = data.qpos[q_adr:q_adr + 3].copy()
    plate_body_pos = data.xpos[plate_body].copy()
    ee_pos = data.site_xpos[ee_site].copy()

    err_qpos = float(np.linalg.norm(plate_qpos_pos - target_pos))
    err_body = float(np.linalg.norm(plate_body_pos - target_pos))
    err_ee = float(np.linalg.norm(ee_pos + CARRY_OFFSET - target_pos))

    print("=== ERROR CALCULATIONS ===")
    print(f"err_qpos (qpos vs target) : {err_qpos:.6f} m")
    print(f"err_body (xpos vs target) : {err_body:.6f} m")
    print(f"err_ee   (ee+carry vs tgt): {err_ee:.6f} m")
    print()

    # ===== INTERPRETATION =====
    print("=== INTERPRETATION ===")
    if nan_in_qpos:
        print("⚠️  NaN detected in qpos → simulation EXPLOSION")
    elif qvel_norm > 100.0:
        print(f"⚠️  qvel_norm={qvel_norm:.1f} (very high) → simulation EXPLOSION")
        print("   Likely cause: gripper↔table residual contacts + long transport")
    elif abs(err_qpos - err_body) > 0.01:
        print(f"⚠️  qpos vs xpos differ by {abs(err_qpos-err_body):.4f} m")
        print("   → possible ID MIXUP (joint id vs body id)")
    elif err_qpos > 1.0:
        print(f"⚠️  plate_qpos_pos={plate_qpos_pos.tolist()}")
        print("   → plate qpos has abnormal values, check q_adr offset")
    else:
        print("✅ Metrics appear consistent. Error is realistic.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
