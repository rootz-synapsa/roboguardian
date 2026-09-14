"""
G3 — Controlled Failure Injector, baseline runs
=================================================

Runs the open-loop baseline (no recovery) pick-and-place trajectory N
times against the same seeded disturbance, and checks whether it fails
reproducibly -- the G3 exit criterion from the build plan:

    "baseline failure reproducible ใน ≥ 8/10 disturbance trials"

This does NOT implement recovery. It is deliberately the dumb baseline:
plan once (HOME -> PRE_GRASP -> GRASP -> LIFT -> PLACE), never re-observe,
never adapt. The disturbance fires once at `--trigger` (default:
before_grasp) and the baseline blindly continues on the stale target.

Failure is defined as: the PLATE's final position misses the place_target
by more than --fail-threshold meters in 3D. This is the correct metric
because the baseline arm blindly reaches the place_target (EE error is
small), but since the plate was moved before grasp, the arm grasps thin
air and the plate is left behind. Task failure = plate is not at target.

Usage:
    python experiments/run_baseline_failure.py \
        --model models/dual_so101_no_table_collision.xml \
        --disturbance object_move \
        --trigger before_grasp \
        --delta 0.10 0.0 0.0 \
        --seed 42 \
        --trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

# Make sibling modules importable when run as `python experiments/run_baseline_failure.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector  # noqa: E402
from evidence.event_logger import EventLogger  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]

# PLACE from clean-scene IK (frozen in G2, 10/10 PASS)
PLACE = [
    -0.0006804320,   # shoulder_pan
    -1.3891631148,   # shoulder_lift
    +1.3904820914,   # elbow_flex
    +1.5105929073,   # wrist_flex
    +0.0039194352,   # wrist_roll
    -0.1745329776,   # gripper (kept from original)
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
FAIL_THRESHOLD_M_DEFAULT = 0.03

# Frozen G2 actuator force limits (evidence-traceable).
# Do not change during G3 validation.
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

    if -1 in (ee_site, target_body, plate_jnt):
        raise LookupError(
            "required model id missing (ee_site/target_body/plate_jnt)"
        )

    # Apply frozen force limits
    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        if act_id < 0:
            raise LookupError(f"actuator not found: {act_name}")
        model.actuator_forcelimited[act_id] = True
        model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, jnt_ids, ee_site, target_body, plate_jnt


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(
    model_path: str,
    trigger: str,
    delta: tuple[float, float, float],
    seed: int,
    fail_threshold_m: float,
    logger: EventLogger,
    trial_index: int,
) -> dict:
    model, act_ids, jnt_ids, ee_site, target_body, plate_jnt = build_model(
        model_path
    )
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    injector = DisturbanceInjector(
        object_joint_name="plate_joint",
        trigger=trigger,
        delta=delta,
        seed=seed + trial_index,
        noise_std=0.0,
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    plate_pose_pre = data.qpos[q_adr:q_adr + 3].copy().tolist()

    # --- before_approach boundary ---
    injector.maybe_fire(model, data, "before_approach")
    for _ in range(100):
        set_arm(data, act_ids, HOME)
        mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP)
        mujoco.mj_step(model, data)

    # --- before_grasp boundary (default disturbance trigger) ---
    injector.maybe_fire(model, data, "before_grasp")
    for _ in range(60):
        set_arm(data, act_ids, GRASP)
        mujoco.mj_step(model, data)

    # --- after_grasp boundary ---
    injector.maybe_fire(model, data, "after_grasp")
    for _ in range(100):
        set_arm(data, act_ids, LIFT)
        mujoco.mj_step(model, data)

    # --- before_transport boundary ---
    injector.maybe_fire(model, data, "before_transport")
    for _ in range(TRANSPORT_STEPS):
        set_arm(data, act_ids, PLACE)
        mujoco.mj_step(model, data)

    # --- before_placement boundary (baseline does NOT re-observe here) ---
    injector.maybe_fire(model, data, "before_placement")

    # ================================================================
    # G3 METRIC: measure the PLATE's final position vs place_target.
    #
    # The baseline arm blindly reaches the place_target (EE error is
    # small ~0.008m), but because the plate was moved before grasp,
    # the arm grasps thin air. The plate is left behind at the
    # disturbed position. Task failure = plate is not at target.
    # ================================================================
    plate_final_pos = data.qpos[q_adr:q_adr + 3].copy()
    target_pos = data.xpos[target_body].copy()

    ee_pos = data.site_xpos[ee_site].copy()
    predicted_release = ee_pos + CARRY_OFFSET
    ee_error_3d = float(np.linalg.norm(predicted_release - target_pos))
    plate_error_3d = float(np.linalg.norm(plate_final_pos - target_pos))

    task_failed = plate_error_3d > fail_threshold_m

    logger.log_step(
        step_id=f"STEP-PLACE-T{trial_index}",
        action_id="ACT-PLACE",
        action="place_plate_baseline",
        expected_pose=plate_pose_pre,
        observed_pose=injector.event.pose_after or plate_pose_pre,
        pose_error=float(np.linalg.norm(
            np.array(injector.event.pose_after or plate_pose_pre)
            - np.array(plate_pose_pre)
        )),
        state="NORMAL",
        decision="EXECUTE",
        executed=True,
        reason="BASELINE_NO_RECOVERY",
        extra={
            "trial_index": trial_index,
            "ee_error_3d_m": ee_error_3d,
            "plate_error_3d_m": plate_error_3d,
            "task_failed": task_failed,
            "disturbance": injector.as_dict(),
        },
    )

    return {
        "trial_index": trial_index,
        "disturbance": injector.as_dict(),
        "target_pose": target_pos.tolist(),
        "plate_final_pose": plate_final_pos.tolist(),
        "predicted_release_pose": predicted_release.tolist(),
        "ee_error_3d_m": ee_error_3d,
        "plate_error_3d_m": plate_error_3d,
        "task_failed": task_failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/dual_so101_no_table_collision.xml",
    )
    parser.add_argument(
        "--disturbance",
        default="object_move",
        help="Label only, for the evidence record.",
    )
    parser.add_argument(
        "--trigger",
        default="before_grasp",
        choices=[
            "before_approach",
            "before_grasp",
            "after_grasp",
            "before_transport",
            "before_placement",
            "after_placement",
        ],
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs=3,
        default=[0.10, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=FAIL_THRESHOLD_M_DEFAULT,
        help="3D distance (m) beyond which the placement counts as failed.",
    )
    parser.add_argument("--out", default="results/g3_failure.json")
    args = parser.parse_args()

    run_id = f"G3-BASELINE-seed{args.seed}"
    logger = EventLogger(run_id=run_id)

    trials = []
    for i in range(args.trials):
        result = run_one_trial(
            model_path=args.model,
            trigger=args.trigger,
            delta=tuple(args.delta),
            seed=args.seed,
            fail_threshold_m=args.fail_threshold,
            logger=logger,
            trial_index=i,
        )
        trials.append(result)
        status = (
            "FAIL (expected)"
            if result["task_failed"]
            else "SUCCESS (unexpected for baseline)"
        )
        print(
            f"trial {i}: "
            f"plate_error_3d={result['plate_error_3d_m']:.4f} m "
            f"(ee_error={result['ee_error_3d_m']:.4f} m) "
            f"-> {status}"
        )

    logger.close()

    n_failed = sum(1 for t in trials if t["task_failed"])
    reproducibility_rate = n_failed / len(trials) if trials else 0.0
    g3_pass = reproducibility_rate >= 0.8  # >= 8/10

    report = {
        "gate": "G3",
        "config": {
            "model": args.model,
            "disturbance_label": args.disturbance,
            "trigger": args.trigger,
            "delta": args.delta,
            "base_seed": args.seed,
            "fail_threshold_m": args.fail_threshold,
            "n_trials": args.trials,
        },
        "trials": trials,
        "n_failed": n_failed,
        "n_total": len(trials),
        "reproducibility_rate": reproducibility_rate,
        "g3_pass": g3_pass,
        "event_log": f"results/events/{run_id}.jsonl",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== G3 {'PASS' if g3_pass else 'FAIL'} ===")
    print(
        f"baseline failed {n_failed}/{len(trials)} trials "
        f"(reproducibility={reproducibility_rate:.0%}, need >= 80%)"
    )
    print(f"Evidence written to: {out_path.resolve()}")
    print(f"Per-step trace: results/events/{run_id}.jsonl")

    return 0 if g3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
