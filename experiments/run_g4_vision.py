"""
G4 (vision variant) — Detection evidence with real OpenVINO perception
==========================================================================

Identical trajectory, disturbance mechanism, and evaluator as
run_detection_check.py. The only change: observe() goes through
VisionStateProvider (the trained + OpenVINO-compiled plate detector)
instead of SimulationStateProvider (ground truth).

This is the evidence that answers section 3's requirement directly --
"OpenVINO ต้องไม่ถูกใส่เพียงเพื่อให้ผ่าน sponsor technology requirement" --
by showing the detection pipeline still works end-to-end when perception
is real instead of ground truth.

IMPORTANT CAVEAT -- check this before trusting the numbers:
The training dataset (dataset_gen.py) rendered frames with the arm at
its RESET pose (mj_resetData, arm never moved). This script renders
frames mid-trajectory (after HOME -> PRE_GRASP), so the arm itself is
in-frame and may occlude part of the workspace in a way the detector
never saw during training. If detection accuracy here is much worse
than the standalone val_mse suggested, this domain gap is the first
thing to check -- not the detector architecture again.

Usage:
    python experiments/run_g4_vision.py \\
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
from control.state_machine import StateEvaluator, StateEvaluatorConfig, NORMAL  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]
PLACE = [
    -0.005315686103320939, -0.4045607705168512, 1.4196,
    0.6715546408917465, 0.0397185882405959, -0.17453297762778586,
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

BOUNDARIES = [
    ("before_approach", HOME, 100),
    (None, PRE_GRASP, 100),
    ("before_grasp", GRASP, 60),
    ("after_grasp", LIFT, 100),
    ("before_transport", PLACE, 2000),
]


def build_model(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)
    act_ids = []
    for name in JOINT_NAMES:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{name}")
        if act_id < 0:
            raise LookupError(f"actuator not found: left_{name}")
        act_ids.append(act_id)
    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    if plate_jnt < 0:
        raise LookupError("joint not found: plate_joint")
    return model, act_ids, plate_jnt


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(
    model_path, onnx_path, norm_path, trigger, delta, seed, trial_index,
    evaluator: StateEvaluator, logger: EventLogger, enable_disturbance: bool,
) -> dict:
    model, act_ids, plate_jnt = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    provider = VisionStateProvider(model, data, onnx_path, norm_path)

    injector = None
    if enable_disturbance:
        injector = DisturbanceInjector("plate_joint", trigger, delta, seed + trial_index, noise_std=0.0)

    boundary_log = []
    detection_success = None
    false_positive = False

    for boundary_name, target, steps in BOUNDARIES:
        for _ in range(steps):
            set_arm(data, act_ids, target)
            mujoco.mj_step(model, data)

        if boundary_name is None:
            continue

        fired_this_step = injector.maybe_fire(model, data, boundary_name) if injector else False

        world_state = provider.observe("plate", expected_pose)
        eval_result = evaluator.evaluate(world_state)

        boundary_log.append({
            "boundary": boundary_name, "fired_this_step": fired_this_step,
            "state": eval_result.state, "reason": eval_result.reason,
            "pose_error": eval_result.pose_error,
            "vision_observed_pose": world_state.observed_pose.tolist(),
        })

        logger.log_step(
            step_id=f"STEP-{boundary_name.upper()}-T{trial_index}",
            action_id=f"ACT-{boundary_name.upper()}", action=boundary_name,
            expected_pose=expected_pose.tolist(), observed_pose=world_state.observed_pose.tolist(),
            pose_error=eval_result.pose_error, state=eval_result.state,
            decision="OBSERVE", executed=False, reason=eval_result.reason,
            extra={"trial_index": trial_index, "disturbed_arm": enable_disturbance, "source": "vision"},
        )

        disturbance_active = injector is not None and injector.event.fired

        if not disturbance_active and eval_result.state != NORMAL:
            false_positive = True

        if disturbance_active and detection_success is None:
            detection_success = eval_result.state != NORMAL

    provider.close()

    return {
        "trial_index": trial_index, "enable_disturbance": enable_disturbance,
        "boundary_log": boundary_log,
        "detected": bool(detection_success) if detection_success is not None else None,
        "false_positive": false_positive,
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
    parser.add_argument("--out", default="results/g4_detection_vision.json")
    args = parser.parse_args()

    evaluator = StateEvaluator(StateEvaluatorConfig(
        normal_threshold_m=args.normal_threshold, recover_threshold_m=args.recover_threshold,
    ))
    logger = EventLogger(run_id=f"G4-VISION-seed{args.seed}")

    disturbed_trials, control_trials = [], []
    for i in range(args.trials):
        disturbed_trials.append(run_one_trial(
            args.model, args.onnx, args.norm, args.trigger, tuple(args.delta),
            args.seed, i, evaluator, logger, enable_disturbance=True,
        ))
    for i in range(args.trials):
        control_trials.append(run_one_trial(
            args.model, args.onnx, args.norm, args.trigger, tuple(args.delta),
            args.seed, i, evaluator, logger, enable_disturbance=False,
        ))

    logger.close()

    n_detected = sum(1 for t in disturbed_trials if t["detected"])
    detection_rate = n_detected / len(disturbed_trials) if disturbed_trials else 0.0
    n_false = sum(1 for t in control_trials if t["false_positive"])
    false_recovery_rate = n_false / len(control_trials) if control_trials else 0.0

    report = {
        "gate": "G4-vision",
        "config": {
            "model": args.model, "onnx": args.onnx, "norm": args.norm,
            "trigger": args.trigger, "delta": args.delta, "seed": args.seed,
            "n_trials": args.trials, "normal_threshold_m": args.normal_threshold,
            "recover_threshold_m": args.recover_threshold,
        },
        "disturbed_trials": disturbed_trials, "control_trials": control_trials,
        "disturbance_detection_rate": detection_rate,
        "false_recovery_rate": false_recovery_rate,
        "g4_vision_pass": detection_rate >= 0.95 and false_recovery_rate <= 0.05,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"disturbance_detection_rate (vision): {detection_rate:.0%} (target >= 95%)")
    print(f"false_recovery_rate (vision):        {false_recovery_rate:.0%} (target <= 5%)")
    print(f"=== G4-vision {'PASS' if report['g4_vision_pass'] else 'FAIL'} ===")
    print(f"Evidence written to: {out_path.resolve()}")
    if not report["g4_vision_pass"]:
        print("\nIf this is much worse than the ground-truth G4 result, check the "
              "training/inference domain gap noted in the module docstring "
              "(arm pose during training vs. during this trajectory) before "
              "assuming the detector itself is the problem.")

    return 0 if report["g4_vision_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
