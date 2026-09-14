"""
G5 (vision variant) — Recovery with real OpenVINO perception
=================================================================

Identical to run_recovery.py's arm C, except RecoveryController is
constructed with a VisionStateProvider instead of letting it default to
ground truth. Both the initial detection AND every re-observation
inside the recovery loop now go through the trained + OpenVINO-compiled
plate detector -- this is the strongest possible evidence that OpenVINO
sits in the actual critical path, not just a side benchmark.

See run_g4_vision.py's docstring for the training/inference domain-gap
caveat (arm pose differs between training renders and mid-trajectory
renders) -- check that first if recovery success rate here is much
worse than the ground-truth G5 result.

Usage:
    python experiments/run_recovery_vision.py \\
        --model models/dual_so101_no_table_collision.xml \\
        --onnx results/plate_detector.onnx \\
        --norm results/plate_detector_norm.json \\
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector, VALID_TRIGGERS  # noqa: E402
from evidence.event_logger import EventLogger  # noqa: E402
from perception.openvino_provider import VisionStateProvider  # noqa: E402
from control.state_machine import StateEvaluator, StateEvaluatorConfig, NORMAL, RECOVERABLE  # noqa: E402
from control.recovery import RecoveryController, ARM_JOINT_NAMES  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]
PLACE = [
    -0.005315686103320939, -0.4045607705168512, 1.4196,
    0.6715546408917465, 0.0397185882405959, -0.17453297762778586,
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CARRY_OFFSET = np.array([0.0, 0.0, 0.048])
TRANSPORT_STEPS = 2000
TASK_FAIL_THRESHOLD_M = 0.03
STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))


def build_model(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)
    act_ids = []
    for name in JOINT_NAMES:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{name}")
        if act_id < 0:
            raise LookupError(f"actuator not found: left_{name}")
        act_ids.append(act_id)
    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
    if -1 in (plate_jnt, ee_site, target_body):
        raise LookupError("required model id missing")
    return model, act_ids, plate_jnt, ee_site, target_body


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(
    model_path, onnx_path, norm_path, trigger, delta, seed, trial_index,
    evaluator: StateEvaluator, logger: EventLogger,
) -> dict:
    model, act_ids, plate_jnt, ee_site, target_body = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    injector = DisturbanceInjector("plate_joint", trigger, delta, seed + trial_index, noise_std=0.0)
    provider = VisionStateProvider(model, data, onnx_path, norm_path)

    for _ in range(100):
        set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)

    injector.maybe_fire(model, data, trigger)

    world_state = provider.observe("plate", expected_pose)
    eval_result = evaluator.evaluate(world_state)

    recovery_trace = None
    safe_stopped = False

    if eval_result.state == NORMAL:
        for _ in range(60):
            set_arm(data, act_ids, GRASP); mujoco.mj_step(model, data)
    elif eval_result.state == RECOVERABLE:
        controller = RecoveryController(
            model, data, act_ids, "left_gripperframe", "plate", evaluator,
            provider=provider,  # <-- the vision provider, reused for every re-observation
        )
        recovery_trace = controller.recover(STALE_GRASP_TARGET, expected_pose)
        logger.log_recovery(
            step_id=f"STEP-RECOVERY-VISION-T{trial_index}", state=recovery_trace.outcome,
            decision="RESUME" if recovery_trace.verified else "SAFE_STOP",
            recovery_attempt=recovery_trace.recovery_attempts_used,
            target_updated=recovery_trace.fresh_target is not None,
            executed=recovery_trace.fresh_action_executed, success=recovery_trace.verified,
            extra={"trial_index": trial_index, "source": "vision", **recovery_trace.to_dict()},
        )
        safe_stopped = not recovery_trace.verified
    else:
        logger.log_safe_stop(
            step_id=f"STEP-RECOVERY-VISION-T{trial_index}",
            reason=f"UNHANDLED_STATE_{eval_result.state}", recovery_attempts_used=0,
        )
        safe_stopped = True

    if safe_stopped:
        provider.close()
        return {
            "trial_index": trial_index, "state_at_grasp": eval_result.state,
            "recovery": recovery_trace.to_dict() if recovery_trace else None,
            "task_failed": True, "safe_stopped": True,
        }

    for _ in range(100):
        set_arm(data, act_ids, LIFT); mujoco.mj_step(model, data)
        data.qpos[q_adr:q_adr + 3] = data.site_xpos[ee_site] + CARRY_OFFSET
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    for _ in range(TRANSPORT_STEPS):
        set_arm(data, act_ids, PLACE); mujoco.mj_step(model, data)
        data.qpos[q_adr:q_adr + 3] = data.site_xpos[ee_site] + CARRY_OFFSET
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    ee_pos = data.site_xpos[ee_site].copy()
    predicted_release = ee_pos + CARRY_OFFSET
    target_pos = data.xpos[target_body].copy()
    error = float(np.linalg.norm(predicted_release - target_pos))
    task_failed = error > TASK_FAIL_THRESHOLD_M

    provider.close()

    return {
        "trial_index": trial_index, "state_at_grasp": eval_result.state,
        "recovery": recovery_trace.to_dict() if recovery_trace else None,
        "target_error_3d_m": error, "task_failed": task_failed, "safe_stopped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_no_table_collision.xml")
    parser.add_argument("--onnx", default="results/plate_detector.onnx")
    parser.add_argument("--norm", default="results/plate_detector_norm.json")
    parser.add_argument("--trigger", default="before_grasp", choices=VALID_TRIGGERS)
    parser.add_argument("--delta", type=float, nargs=3, default=[0.10, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--normal-threshold", type=float, default=0.01)
    parser.add_argument("--recover-threshold", type=float, default=0.15)
    parser.add_argument("--out", default="results/g5_recovery_vision.json")
    args = parser.parse_args()

    evaluator = StateEvaluator(StateEvaluatorConfig(
        normal_threshold_m=args.normal_threshold, recover_threshold_m=args.recover_threshold,
    ))
    logger = EventLogger(run_id=f"G5-VISION-seed{args.seed}")

    trials = []
    for i in range(args.trials):
        result = run_one_trial(
            args.model, args.onnx, args.norm, args.trigger, tuple(args.delta),
            args.seed, i, evaluator, logger,
        )
        trials.append(result)
        status = "SUCCESS" if not result["task_failed"] else ("SAFE_STOP" if result["safe_stopped"] else "FAIL")
        err = result.get("target_error_3d_m")
        err_str = f"error={err:.4f}m" if err is not None else "error=n/a"
        print(f"trial {i}: state={result['state_at_grasp']} {err_str} -> {status}")

    logger.close()

    n_success = sum(1 for t in trials if not t["task_failed"])
    recovery_success_rate = n_success / len(trials) if trials else 0.0
    stale_never_executed = all(
        (t["recovery"] is None) or (t["recovery"]["stale_action_executed"] is False)
        for t in trials
    )

    report = {
        "gate": "G5-vision",
        "config": {
            "model": args.model, "onnx": args.onnx, "norm": args.norm,
            "trigger": args.trigger, "delta": args.delta, "seed": args.seed, "n_trials": args.trials,
            "normal_threshold_m": args.normal_threshold, "recover_threshold_m": args.recover_threshold,
        },
        "trials": trials,
        "recovery_success_rate": recovery_success_rate,
        "stale_target_never_executed_across_all_trials": stale_never_executed,
        "g5_vision_pass": recovery_success_rate >= 0.80 and stale_never_executed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nrecovery_success_rate (vision): {recovery_success_rate:.0%} (target >= 80%)")
    print(f"stale target never executed:    {stale_never_executed}")
    print(f"=== G5-vision {'PASS' if report['g5_vision_pass'] else 'FAIL'} ===")
    print(f"Evidence written to: {out_path.resolve()}")

    return 0 if report["g5_vision_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
