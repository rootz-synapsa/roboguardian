"""
G5 — Recovery evidence runner (Arm C: RoboGuardian).

Same trajectory and same seeded disturbance as G3/G4. The difference is
what happens when the state evaluator returns RECOVERABLE at the
before_grasp boundary: instead of continuing with the stale GRASP joint
target (the baseline behavior measured in G3), this hands off to
RecoveryController, which re-observes, re-solves a fresh grasp target,
executes only that, and verifies the grasp before resuming.

Provider modes:
  --provider simulation   Use SimulationStateProvider (ground truth)
  --provider vision       Use VisionStateProvider (OpenVINO inference)

NOTE: RecoveryController internally uses SimulationStateProvider for
its re-observation step (ground truth). This means the initial
detection uses the selected provider, but the recovery re-observation
always uses ground truth. For a full end-to-end vision test,
RecoveryController would need to accept a provider parameter.

Evidence written to results/g5_recovery.json must let a reviewer
directly check the section 18 claim:
  stale target = A          (trace.stale_target)
  fresh target = B          (trace.fresh_target)
  action against A = never issued   (trace.stale_action_executed == False)
  action against B = executed       (trace.fresh_action_executed == True)

Usage:
    # Ground truth mode (G5 original):
    python experiments/run_recovery.py \
        --model models/dual_so101_g3.xml \
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10 \
        --provider simulation

    # Vision mode (G7 integration test):
    python experiments/run_recovery.py \
        --model models/dual_so101_g3.xml \
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10 \
        --provider vision \
        --normal-threshold 0.03 \
        --onnx-model results/plate_detector.onnx \
        --norm-stats results/plate_detector_norm.json \
        --out results/g5_recovery_vision.json
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
    RECOVERABLE,
)
from control.recovery import RecoveryController, ARM_JOINT_NAMES  # noqa: E402

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
TASK_FAIL_THRESHOLD_M = 0.05  # consistent with G2 PLACE_XY_TOLERANCE

# Frozen G2 force limits
FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}

STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))


def build_provider(provider_mode: str, model, data, args):
    """Factory: returns a StateProvider instance based on --provider flag."""
    if provider_mode == "simulation":
        from perception.state_provider import SimulationStateProvider
        return SimulationStateProvider(model, data)

    elif provider_mode == "vision":
        from perception.openvino_provider import VisionStateProvider
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
    ee_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe"
    )
    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "place_target"
    )

    if -1 in (plate_jnt, ee_site, target_body):
        raise LookupError("required model id missing")

    # Apply frozen force limits
    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, plate_jnt, ee_site, target_body


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(
    model_path: str,
    trigger: str,
    delta: tuple[float, float, float],
    seed: int,
    trial_index: int,
    evaluator: StateEvaluator,
    logger: EventLogger,
    provider_mode: str,
    args,
) -> dict:
    model, act_ids, plate_jnt, ee_site, target_body = build_model(
        model_path
    )
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    injector = DisturbanceInjector(
        object_joint_name="plate_joint",
        trigger=trigger,
        delta=delta,
        seed=seed + trial_index,
        noise_std=0.0,
    )

    provider = build_provider(provider_mode, model, data, args)

    # Approach
    for _ in range(100):
        set_arm(data, act_ids, HOME)
        mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP)
        mujoco.mj_step(model, data)

    # Disturbance fires here
    injector.maybe_fire(model, data, "before_grasp")

    world_state = provider.observe("plate", expected_pose)
    eval_result = evaluator.evaluate(world_state)

    recovery_trace = None
    safe_stopped = False

    if eval_result.state == NORMAL:
        for _ in range(60):
            set_arm(data, act_ids, GRASP)
            mujoco.mj_step(model, data)

    elif eval_result.state == RECOVERABLE:
        # NOTE: RecoveryController internally uses SimulationStateProvider
        # (ground truth) for its re-observation step. The initial detection
        # above uses the selected provider (simulation or vision).
        controller = RecoveryController(
            model=model,
            data=data,
            act_ids=act_ids,
            ee_site_name="left_gripperframe",
            object_body_name="plate",
            evaluator=evaluator,
        )
        recovery_trace = controller.recover(
            STALE_GRASP_TARGET, expected_pose
        )

        logger.log_recovery(
            step_id=f"STEP-RECOVERY-T{trial_index}",
            state=recovery_trace.outcome,
            decision=(
                "RESUME" if recovery_trace.verified else "SAFE_STOP"
            ),
            recovery_attempt=recovery_trace.recovery_attempts_used,
            target_updated=recovery_trace.fresh_target is not None,
            executed=recovery_trace.fresh_action_executed,
            success=recovery_trace.verified,
            extra={
                "trial_index": trial_index,
                "provider": provider_mode,
                **recovery_trace.to_dict(),
            },
        )

        if not recovery_trace.verified:
            safe_stopped = True

    else:
        logger.log_safe_stop(
            step_id=f"STEP-RECOVERY-T{trial_index}",
            reason=f"UNHANDLED_STATE_{eval_result.state}",
            recovery_attempts_used=0,
        )
        safe_stopped = True

    # Clean up vision provider if applicable
    if hasattr(provider, "close"):
        provider.close()

    if safe_stopped:
        return {
            "trial_index": trial_index,
            "state_at_grasp": eval_result.state,
            "recovery": (
                recovery_trace.to_dict() if recovery_trace else None
            ),
            "task_failed": True,
            "safe_stopped": True,
        }

    # Lift while carrying
    for _ in range(100):
        set_arm(data, act_ids, LIFT)
        mujoco.mj_step(model, data)
        data.qpos[q_adr:q_adr + 3] = (
            data.site_xpos[ee_site] + CARRY_OFFSET
        )
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    # Transport while carrying
    for _ in range(TRANSPORT_STEPS):
        set_arm(data, act_ids, PLACE)
        mujoco.mj_step(model, data)
        data.qpos[q_adr:q_adr + 3] = (
            data.site_xpos[ee_site] + CARRY_OFFSET
        )
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    # Score placement (XY error, consistent with G2)
    ee_pos = data.site_xpos[ee_site].copy()
    predicted_release = ee_pos + CARRY_OFFSET
    target_pos = data.xpos[target_body].copy()

    target_error_3d = float(
        np.linalg.norm(predicted_release - target_pos)
    )
    target_error_xy = float(
        np.linalg.norm(predicted_release[:2] - target_pos[:2])
    )

    task_failed = target_error_xy > TASK_FAIL_THRESHOLD_M

    logger.log_step(
        step_id=f"STEP-PLACE-T{trial_index}",
        action_id="ACT-PLACE",
        action="place_plate_recovered",
        expected_pose=expected_pose.tolist(),
        observed_pose=target_pos.tolist(),
        pose_error=target_error_xy,
        state=eval_result.state,
        decision=(
            "RESUME_AFTER_RECOVERY" if recovery_trace else "EXECUTE"
        ),
        executed=True,
        extra={
            "trial_index": trial_index,
            "provider": provider_mode,
            "target_error_3d_m": target_error_3d,
            "target_error_xy_m": target_error_xy,
        },
    )

    return {
        "trial_index": trial_index,
        "state_at_grasp": eval_result.state,
        "recovery": (
            recovery_trace.to_dict() if recovery_trace else None
        ),
        "target_error_3d_m": target_error_3d,
        "target_error_xy_m": target_error_xy,
        "task_failed": task_failed,
        "safe_stopped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="models/dual_so101_g3.xml"
    )
    parser.add_argument(
        "--trigger", default="before_grasp", choices=VALID_TRIGGERS
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
    parser.add_argument("--out", default="results/g5_recovery.json")
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
        run_id=f"G5-RECOVERY-{run_id_suffix}-seed{args.seed}"
    )

    trials = []
    for i in range(args.trials):
        result = run_one_trial(
            args.model,
            args.trigger,
            tuple(args.delta),
            args.seed,
            i,
            evaluator,
            logger,
            provider_mode=args.provider,
            args=args,
        )
        trials.append(result)

        status = (
            "SUCCESS"
            if not result["task_failed"]
            else (
                "SAFE_STOP" if result["safe_stopped"] else "FAIL"
            )
        )
        err = result.get("target_error_xy_m")
        err_str = (
            f"xy_error={err:.4f}m" if err is not None else "error=n/a"
        )
        print(
            f"trial {i}: "
            f"state={result['state_at_grasp']} "
            f"{err_str} -> {status}"
        )

    logger.close()

    n_success = sum(1 for t in trials if not t["task_failed"])
    recovery_success_rate = (
        n_success / len(trials) if trials else 0.0
    )

    stale_never_executed = all(
        (t["recovery"] is None)
        or (t["recovery"]["stale_action_executed"] is False)
        for t in trials
    )

    report = {
        "gate": "G5",
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
        "trials": trials,
        "recovery_success_rate": recovery_success_rate,
        "stale_target_never_executed_across_all_trials": (
            stale_never_executed
        ),
        "g5_pass": (
            recovery_success_rate >= 0.80 and stale_never_executed
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nprovider:                   {args.provider}")
    print(
        f"recovery_success_rate: "
        f"{recovery_success_rate:.0%} (target >= 80%)"
    )
    print(f"stale target never executed: {stale_never_executed}")
    print(
        f"=== G5 {'PASS' if report['g5_pass'] else 'FAIL'} ==="
    )
    print(f"Evidence written to: {out_path.resolve()}")

    return 0 if report["g5_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
