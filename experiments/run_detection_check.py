"""
G4 — Detection evidence runner

Reuses the exact G3 trajectory/disturbance mechanism, but instead of
just measuring final placement error, it calls the REAL
StateProvider + StateEvaluator at every trigger boundary and checks
whether the classification matches what should happen:

  - before the disturbance fires: state must be NORMAL
  - at/after the boundary where it fires: state must NOT be NORMAL
    (RECOVERABLE is what we expect for this delta magnitude)

It also runs an UNDISTURBED control arm (same trajectory, no injector
firing) to measure the False Recovery Rate — i.e. does the observer
cry wolf when nothing happened.

Provider modes:
  --provider simulation   Use SimulationStateProvider (ground truth from MuJoCo)
  --provider vision       Use VisionStateProvider (OpenVINO inference)

Metrics written to results/g4_detection.json:
  disturbance_detection_rate  (target >= 95%)
  false_recovery_rate         (target <= 5%)

Usage:
    # Ground truth mode (G4 original):
    python experiments/run_detection_check.py \
        --model models/dual_so101_g3.xml \
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10 \
        --provider simulation

    # Vision mode (G7 integration test):
    python experiments/run_detection_check.py \
        --model models/dual_so101_g3.xml \
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10 \
        --provider vision \
        --onnx-model results/plate_detector.onnx \
        --norm-stats results/plate_detector_norm.json \
        --out results/g4_detection_vision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector, VALID_TRIGGERS  # noqa: E402
from evidence.event_logger import EventLogger  # noqa: E402
from control.state_machine import (  # noqa: E402
    StateEvaluator,
    StateEvaluatorConfig,
    NORMAL,
)

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

# Frozen G2 force limits
FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}

BOUNDARIES = [
    ("before_approach", HOME, 100),
    (None, PRE_GRASP, 100),
    ("before_grasp", GRASP, 60),
    ("after_grasp", LIFT, 100),
    ("before_transport", PLACE, 2000),
]


def build_provider(provider_mode: str, model, data, args):
    """Factory: returns a StateProvider instance based on --provider flag."""
    if provider_mode == "simulation":
        from perception.state_provider import SimulationStateProvider
        return SimulationStateProvider(model, data)

    elif provider_mode == "vision":
        from perception.openvino_provider import VisionStateProvider
        # Use partial to pre-fill onnx_model_path and norm_stats_path
        # so the call signature matches SimulationStateProvider(model, data)
        VisionProviderWithDefaults = partial(
            VisionStateProvider,
            onnx_model_path=args.onnx_model,
            norm_stats_path=args.norm_stats,
        )
        return VisionProviderWithDefaults(model, data)

    else:
        raise ValueError(f"Unknown provider mode: {provider_mode}")


def build_model(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)

    act_ids = []
    for name in JOINT_NAMES:
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{name}"
        )
        if act_id < 0:
            raise LookupError(f"actuator not found: left_{name}")
        act_ids.append(act_id)

    plate_jnt = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint"
    )
    if plate_jnt < 0:
        raise LookupError("joint not found: plate_joint")

    # Apply frozen force limits
    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, plate_jnt


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(
    model_path: str,
    trigger: str | None,
    delta: tuple[float, float, float],
    seed: int,
    trial_index: int,
    evaluator: StateEvaluator,
    logger: EventLogger,
    enable_disturbance: bool,
    provider_mode: str,
    args,
) -> dict:
    model, act_ids, plate_jnt = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    provider = build_provider(provider_mode, model, data, args)

    injector = None
    if enable_disturbance:
        injector = DisturbanceInjector(
            object_joint_name="plate_joint",
            trigger=trigger,
            delta=delta,
            seed=seed + trial_index,
            noise_std=0.0,
        )

    boundary_log = []
    detection_success = None
    false_positive = False

    for boundary_name, target, steps in BOUNDARIES:
        for _ in range(steps):
            set_arm(data, act_ids, target)
            mujoco.mj_step(model, data)

        if boundary_name is None:
            continue

        fired_this_step = False
        if injector is not None:
            fired_this_step = injector.maybe_fire(
                model, data, boundary_name
            )

        world_state = provider.observe("plate", expected_pose)
        eval_result = evaluator.evaluate(world_state)

        boundary_log.append({
            "boundary": boundary_name,
            "fired_this_step": fired_this_step,
            "state": eval_result.state,
            "reason": eval_result.reason,
            "pose_error": eval_result.pose_error,
        })

        logger.log_step(
            step_id=f"STEP-{boundary_name.upper()}-T{trial_index}",
            action_id=f"ACT-{boundary_name.upper()}",
            action=boundary_name,
            expected_pose=expected_pose.tolist(),
            observed_pose=world_state.observed_pose.tolist(),
            pose_error=eval_result.pose_error,
            state=eval_result.state,
            decision="OBSERVE",
            executed=False,
            reason=eval_result.reason,
            extra={
                "trial_index": trial_index,
                "disturbed_arm": enable_disturbance,
                "provider": provider_mode,
            },
        )

        disturbance_active = (
            injector is not None and injector.event.fired
        )

        if not disturbance_active and eval_result.state != NORMAL:
            false_positive = True

        if disturbance_active and detection_success is None:
            detection_success = eval_result.state != NORMAL

    # Clean up vision provider if applicable
    if hasattr(provider, "close"):
        provider.close()

    return {
        "trial_index": trial_index,
        "enable_disturbance": enable_disturbance,
        "boundary_log": boundary_log,
        "detected": (
            bool(detection_success)
            if detection_success is not None
            else None
        ),
        "false_positive": false_positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/dual_so101_g3.xml",
    )
    parser.add_argument(
        "--trigger",
        default="before_grasp",
        choices=VALID_TRIGGERS,
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
        "--normal-threshold", type=float, default=0.01
    )
    parser.add_argument(
        "--recover-threshold", type=float, default=0.15
    )
    parser.add_argument(
        "--provider",
        default="simulation",
        choices=["simulation", "vision"],
        help=(
            "simulation = ground truth from MuJoCo, "
            "vision = OpenVINO inference"
        ),
    )
    parser.add_argument(
        "--onnx-model",
        default="results/plate_detector.onnx",
        help="Path to ONNX model (only used with --provider vision)",
    )
    parser.add_argument(
        "--norm-stats",
        default="results/plate_detector_norm.json",
        help=(
            "Path to normalization stats JSON "
            "(only used with --provider vision)"
        ),
    )
    parser.add_argument(
        "--out", default="results/g4_detection.json"
    )
    args = parser.parse_args()

    evaluator = StateEvaluator(
        StateEvaluatorConfig(
            normal_threshold_m=args.normal_threshold,
            recover_threshold_m=args.recover_threshold,
        )
    )

    run_id_suffix = (
        "vision" if args.provider == "vision" else "simulation"
    )
    logger = EventLogger(
        run_id=f"G4-DETECTION-{run_id_suffix}-seed{args.seed}"
    )

    disturbed_trials = []
    for i in range(args.trials):
        disturbed_trials.append(
            run_one_trial(
                args.model,
                args.trigger,
                tuple(args.delta),
                args.seed,
                i,
                evaluator,
                logger,
                enable_disturbance=True,
                provider_mode=args.provider,
                args=args,
            )
        )

    control_trials = []
    for i in range(args.trials):
        control_trials.append(
            run_one_trial(
                args.model,
                args.trigger,
                tuple(args.delta),
                args.seed,
                i,
                evaluator,
                logger,
                enable_disturbance=False,
                provider_mode=args.provider,
                args=args,
            )
        )

    logger.close()

    n_detected = sum(
        1 for t in disturbed_trials if t["detected"]
    )
    detection_rate = (
        n_detected / len(disturbed_trials)
        if disturbed_trials
        else 0.0
    )

    n_false = sum(
        1 for t in control_trials if t["false_positive"]
    )
    false_recovery_rate = (
        n_false / len(control_trials)
        if control_trials
        else 0.0
    )

    report = {
        "gate": "G4",
        "provider": args.provider,
        "config": {
            "model": args.model,
            "trigger": args.trigger,
            "delta": args.delta,
            "seed": args.seed,
            "n_trials": args.trials,
            "normal_threshold_m": args.normal_threshold,
            "recover_threshold_m": args.recover_threshold,
            "onnx_model": (
                args.onnx_model
                if args.provider == "vision"
                else None
            ),
        },
        "disturbed_trials": disturbed_trials,
        "control_trials": control_trials,
        "disturbance_detection_rate": detection_rate,
        "false_recovery_rate": false_recovery_rate,
        "g4_pass": (
            detection_rate >= 0.95
            and false_recovery_rate <= 0.05
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"provider:                   {args.provider}"
    )
    print(
        f"disturbance_detection_rate: "
        f"{detection_rate:.0%} (target >= 95%)"
    )
    print(
        f"false_recovery_rate:        "
        f"{false_recovery_rate:.0%} (target <= 5%)"
    )
    print(
        f"=== G4 {'PASS' if report['g4_pass'] else 'FAIL'} ==="
    )
    print(f"Evidence written to: {out_path.resolve()}")

    return 0 if report["g4_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
