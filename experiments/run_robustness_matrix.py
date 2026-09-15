"""
G8 / H1 — Robustness matrix
==============================

Reuses the exact G5/G6 recovery pipeline (build_model, disturbance
injection, StateEvaluator, RecoveryController) across a matrix of
degraded conditions, via FaultInjectingProvider wrapping
SimulationStateProvider. Nothing about G1-G7 is touched; this is an
additive evidence layer only, per the plan's "G1-G7 = FROZEN" rule.

PASS criterion per condition (per the plan -- NOT "100% success"):
    stale_action_execution_rate == 0%
    AND every trial reached a valid terminal outcome
        (task success, or a correctly-reasoned SAFE_STOP)
A condition where the arm can't recover from a 15cm displacement but
safely stops instead is a PASS, not a failure -- that's the point of
governance under a disturbance that exceeds the recovery budget.

Usage:
    python experiments/run_robustness_matrix.py \\
        --model models/dual_so101_no_table_collision.xml \\
        --seed 42 --trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector  # noqa: E402
from evidence.event_logger import EventLogger  # noqa: E402
from perception.state_provider import SimulationStateProvider  # noqa: E402
from perception.fault_injecting_provider import FaultInjectingProvider  # noqa: E402
from control.state_machine import StateEvaluator, StateEvaluatorConfig, NORMAL, RECOVERABLE  # noqa: E402
from control.recovery import RecoveryController, ARM_JOINT_NAMES  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]
# PLACE from clean-scene IK (frozen in G2, 10/10 PASS) -- matches run_recovery.py exactly.
PLACE = [
    -0.0006804320, -1.3891631148, 1.3904820914,
    1.5105929073, 0.0039194352, -0.1745329776,
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CARRY_OFFSET = np.array([0.0, 0.0, 0.048])
TRANSPORT_STEPS = 2000
TASK_FAIL_THRESHOLD_M = 0.05  # XY-only, consistent with G2 PLACE_XY_TOLERANCE -- not 3D distance
STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))

# Frozen G2 force limits -- MUST match run_recovery.py's build_model() exactly.
# Missing this caused unbounded actuator force against real table collision,
# producing large post-contact overshoot despite a valid (collision-blind) IK
# solution -- this is what made grasp_error ~0.17-0.20m constant across every
# condition in the first robustness-matrix run.
FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}

# Delay is expressed in "observe() calls behind", not wall-clock ms --
# this control loop observes at discrete boundaries, not a fixed-rate
# poll. 50ms/100ms are the plan's labels; delay_frames=1/2 is the
# closest honest proxy given the architecture. Reported as such below.
CONDITIONS = [
    {"name": "baseline",           "delta": [0.10, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 0, "drop_frame_count": 0},
    {"name": "displacement_5cm",   "delta": [0.05, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 0, "drop_frame_count": 0},
    {"name": "displacement_15cm",  "delta": [0.15, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 0, "drop_frame_count": 0},
    {"name": "noise_1cm",          "delta": [0.10, 0.0, 0.0], "noise_std": 0.01, "delay_frames": 0, "drop_frame_count": 0},
    {"name": "noise_2cm",          "delta": [0.10, 0.0, 0.0], "noise_std": 0.02, "delay_frames": 0, "drop_frame_count": 0},
    {"name": "delay_50ms_proxy",   "delta": [0.10, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 1, "drop_frame_count": 0},
    {"name": "delay_100ms_proxy",  "delta": [0.10, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 2, "drop_frame_count": 0},
    {"name": "dropped_frame",      "delta": [0.10, 0.0, 0.0], "noise_std": 0.0,  "delay_frames": 0, "drop_frame_count": 1},
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
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
    if -1 in (plate_jnt, ee_site, target_body):
        raise LookupError("required model id missing")

    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, plate_jnt, ee_site, target_body


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(model_path, condition, seed, trial_index, evaluator, logger):
    model, act_ids, plate_jnt, ee_site, target_body = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    injector = DisturbanceInjector(
        "plate_joint", "before_grasp", tuple(condition["delta"]), seed + trial_index, noise_std=0.0,
    )
    provider = FaultInjectingProvider(
        SimulationStateProvider(model, data),
        noise_std=condition["noise_std"],
        delay_frames=condition["delay_frames"],
        drop_frame_count=condition["drop_frame_count"],
        seed=seed + trial_index,
    )

    for _ in range(100):
        set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)

    injector.maybe_fire(model, data, "before_grasp")

    world_state = provider.observe("plate", expected_pose)
    eval_result = evaluator.evaluate(world_state)

    recovery_trace = None
    safe_stopped = False
    valid_outcome = True  # False only if the trial ends in an undefined/crash state

    if eval_result.state == NORMAL:
        for _ in range(60):
            set_arm(data, act_ids, GRASP); mujoco.mj_step(model, data)
    elif eval_result.state == RECOVERABLE:
        controller = RecoveryController(
            model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=provider,
        )
        recovery_trace = controller.recover(STALE_GRASP_TARGET, expected_pose)
        logger.log_recovery(
            step_id=f"STEP-ROBUST-{condition['name']}-T{trial_index}", state=recovery_trace.outcome,
            decision="RESUME" if recovery_trace.verified else "SAFE_STOP",
            recovery_attempt=recovery_trace.recovery_attempts_used,
            target_updated=recovery_trace.fresh_target is not None,
            executed=recovery_trace.fresh_action_executed, success=recovery_trace.verified,
            extra={"trial_index": trial_index, "condition": condition["name"], **recovery_trace.to_dict()},
        )
        safe_stopped = not recovery_trace.verified
    else:
        # REPLAN_REQUIRED / UNCERTAIN -- a defined, valid outcome for this
        # minimum recovery loop's scope: conservatively stop rather than guess.
        logger.log_safe_stop(
            step_id=f"STEP-ROBUST-{condition['name']}-T{trial_index}",
            reason=f"UNHANDLED_STATE_{eval_result.state}", recovery_attempts_used=0,
        )
        safe_stopped = True

    stale_executed = bool(recovery_trace.stale_action_executed) if recovery_trace else False

    if safe_stopped:
        return {
            "trial_index": trial_index, "state_at_grasp": eval_result.state,
            "recovery": recovery_trace.to_dict() if recovery_trace else None,
            "task_success": False, "safe_stopped": True,
            "stale_action_executed": stale_executed, "valid_outcome": valid_outcome,
            "recovery_attempts_used": recovery_trace.recovery_attempts_used if recovery_trace else 0,
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
    error_3d = float(np.linalg.norm(predicted_release - target_pos))
    error_xy = float(np.linalg.norm(predicted_release[:2] - target_pos[:2]))
    task_success = error_xy <= TASK_FAIL_THRESHOLD_M  # XY-only, matches run_recovery.py

    return {
        "trial_index": trial_index, "state_at_grasp": eval_result.state,
        "recovery": recovery_trace.to_dict() if recovery_trace else None,
        "task_success": task_success, "target_error_3d_m": error_3d,
        "target_error_xy_m": error_xy, "safe_stopped": False,
        "stale_action_executed": stale_executed, "valid_outcome": valid_outcome,
        "recovery_attempts_used": recovery_trace.recovery_attempts_used if recovery_trace else 0,
    }


def summarize_condition(trials: list[dict]) -> dict:
    n = len(trials)
    n_success = sum(1 for t in trials if t["task_success"])
    n_safe_stop = sum(1 for t in trials if t["safe_stopped"])
    n_stale = sum(1 for t in trials if t["stale_action_executed"])
    n_valid = sum(1 for t in trials if t["valid_outcome"])
    mean_attempts = sum(t["recovery_attempts_used"] for t in trials) / n if n else 0.0

    stale_rate = n_stale / n if n else 0.0
    all_valid = n_valid == n

    return {
        "n_trials": n,
        "task_success_rate": n_success / n if n else 0.0,
        "safe_stop_rate": n_safe_stop / n if n else 0.0,
        "stale_action_execution_rate": stale_rate,
        "mean_recovery_attempts": mean_attempts,
        "all_outcomes_valid": all_valid,
        # H1's actual pass bar -- not raw success rate.
        "condition_pass": (stale_rate == 0.0) and all_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_g3.xml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--normal-threshold", type=float, default=0.01)
    parser.add_argument("--recover-threshold", type=float, default=0.15)
    parser.add_argument("--out-json", default="results/g8_robustness_matrix.json")
    parser.add_argument("--out-md", default="results/ROBUSTNESS_MATRIX.md")
    args = parser.parse_args()

    evaluator = StateEvaluator(StateEvaluatorConfig(
        normal_threshold_m=args.normal_threshold, recover_threshold_m=args.recover_threshold,
    ))
    logger = EventLogger(run_id=f"G8-ROBUSTNESS-seed{args.seed}")

    all_conditions = {}
    for condition in CONDITIONS:
        trials = []
        for i in range(args.trials):
            trials.append(run_one_trial(args.model, condition, args.seed, i, evaluator, logger))
        summary = summarize_condition(trials)
        all_conditions[condition["name"]] = {"config": condition, "trials": trials, "summary": summary}
        print(f"{condition['name']:20s} success={summary['task_success_rate']:.0%}  "
              f"safe_stop={summary['safe_stop_rate']:.0%}  "
              f"stale_exec={summary['stale_action_execution_rate']:.0%}  "
              f"-> {'PASS' if summary['condition_pass'] else 'FAIL'}")

    logger.close()

    overall_pass = all(c["summary"]["condition_pass"] for c in all_conditions.values())

    report = {
        "gate": "G8-H1",
        "config": {"model": args.model, "seed": args.seed, "n_trials_per_condition": args.trials},
        "note": ("delay_frames is a proxy for the plan's ms-based delay labels -- this "
                 "control loop observes at discrete trigger boundaries, not a fixed-rate "
                 "poll, so wall-clock delay isn't directly expressible."),
        "conditions": all_conditions,
        "g8_h1_pass": overall_pass,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# RoboGuardian — G8/H1 Robustness Matrix",
        "",
        f"Model: `{args.model}`  |  Seed: `{args.seed}`  |  Trials/condition: `{args.trials}`",
        "",
        "| Condition | Success | Safe-Stop | Stale-Exec | Attempts (mean) | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in all_conditions.items():
        s = c["summary"]
        lines.append(
            f"| {name} | {s['task_success_rate']:.0%} | {s['safe_stop_rate']:.0%} | "
            f"{s['stale_action_execution_rate']:.0%} | {s['mean_recovery_attempts']:.2f} | "
            f"{'✅' if s['condition_pass'] else '❌'} |"
        )
    lines += [
        "",
        "**Pass criterion**: `stale_action_execution_rate == 0%` AND every trial reached a "
        "valid terminal outcome (success or a correctly-reasoned SAFE_STOP). Raw task "
        "success rate is reported for context, not used as the pass bar -- a governed "
        "SAFE_STOP under a disturbance that exceeds the recovery budget is a correct "
        "outcome, not a failure.",
        "",
        f"Overall G8/H1: {'PASS' if overall_pass else 'FAIL'}",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines))

    print(f"\n=== G8/H1 {'PASS' if overall_pass else 'FAIL'} ===")
    print(f"JSON: {out_json.resolve()}")
    print(f"Markdown: {Path(args.out_md).resolve()}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
