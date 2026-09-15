"""
G8 / H2 — End-to-end control-loop latency budget
====================================================

Answers: from "the world changed" to "a fresh, authorized action is
issued", how long does that actually take?

Reuses the same recovery pipeline as G5/G6/H1 (same build_model with
frozen G2 force limits, same PLACE, same StateEvaluator/
RecoveryController) -- this benchmark adds NO new behavior, only
timing instrumentation, per the H2 rule "อย่ากำหนด speed claim
ล่วงหน้า". Whatever the numbers say is what gets reported.

Stages measured:
  - perception_ms / evaluation_ms   : the INITIAL observe+evaluate at
                                       before_grasp, before recovery is
                                       even decided (this is the same
                                       quantity G6 called
                                       decision_latency_ms).
  - replan_ms                       : summed solve_arm_ik() time across
                                       every recovery attempt in the trial.
  - observation_ms / evaluation_ms
    (recovery-internal)             : summed re-observation time inside
                                       RecoveryController.recover().
  - execution_ms                    : summed _execute_arm() time.
  - observe_to_decision_ms          : perception_ms + evaluation_ms
                                       (initial stage only) -- the
                                       "did the world change" latency.
  - recovery_total_ms               : the whole recover() call, wall clock.

Usage:
    python experiments/benchmark_control_loop.py \\
        --model models/dual_so101_g3.xml \\
        --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.disturbance import DisturbanceInjector, VALID_TRIGGERS  # noqa: E402
from evidence.event_logger import EventLogger  # noqa: E402
from perception.state_provider import SimulationStateProvider  # noqa: E402
from control.state_machine import StateEvaluator, StateEvaluatorConfig, NORMAL, RECOVERABLE  # noqa: E402
from control.recovery import RecoveryController, ARM_JOINT_NAMES  # noqa: E402

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))

# Frozen G2 force limits -- must match run_recovery.py / run_robustness_matrix.py.
FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}


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

    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]

    return model, act_ids, plate_jnt


def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])


def run_one_trial(model_path, trigger, delta, seed, trial_index, evaluator, logger):
    model, act_ids, plate_jnt = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    injector = DisturbanceInjector("plate_joint", trigger, delta, seed + trial_index, noise_std=0.0)
    provider = SimulationStateProvider(model, data)

    for _ in range(100):
        set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100):
        set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)

    injector.maybe_fire(model, data, trigger)

    # Initial detection stage -- this is the "did the world change" latency,
    # measured BEFORE any recovery decision is made.
    t0 = time.perf_counter()
    world_state = provider.observe("plate", expected_pose)
    t1 = time.perf_counter()
    eval_result = evaluator.evaluate(world_state)
    t2 = time.perf_counter()
    initial_perception_ms = (t1 - t0) * 1000.0
    initial_evaluation_ms = (t2 - t1) * 1000.0

    if eval_result.state != RECOVERABLE:
        # Not the scenario this benchmark targets -- log what we have and move on.
        return {
            "trial_index": trial_index, "state_at_grasp": eval_result.state,
            "initial_perception_ms": initial_perception_ms,
            "initial_evaluation_ms": initial_evaluation_ms,
            "observe_to_decision_ms": initial_perception_ms + initial_evaluation_ms,
            "recovery": None,
        }

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=provider,
    )
    recovery_trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    logger.log_recovery(
        step_id=f"STEP-LATENCY-T{trial_index}", state=recovery_trace.outcome,
        decision="RESUME" if recovery_trace.verified else "SAFE_STOP",
        recovery_attempt=recovery_trace.recovery_attempts_used,
        target_updated=recovery_trace.fresh_target is not None,
        executed=recovery_trace.fresh_action_executed, success=recovery_trace.verified,
        extra={"trial_index": trial_index, **recovery_trace.to_dict()},
    )

    total_replan_ms = sum(a.get("timing_ms", {}).get("replan_ms", 0.0) for a in recovery_trace.attempts_log)
    total_recovery_observation_ms = sum(
        a.get("timing_ms", {}).get("observation_ms", 0.0) for a in recovery_trace.attempts_log
    )
    total_recovery_evaluation_ms = sum(
        a.get("timing_ms", {}).get("evaluation_ms", 0.0) for a in recovery_trace.attempts_log
    )
    total_execution_ms = sum(a.get("timing_ms", {}).get("execution_ms", 0.0) for a in recovery_trace.attempts_log)

    return {
        "trial_index": trial_index, "state_at_grasp": eval_result.state,
        "initial_perception_ms": initial_perception_ms,
        "initial_evaluation_ms": initial_evaluation_ms,
        "observe_to_decision_ms": initial_perception_ms + initial_evaluation_ms,
        "recovery": {
            "verified": recovery_trace.verified,
            "recovery_attempts_used": recovery_trace.recovery_attempts_used,
            "recovery_observation_ms": total_recovery_observation_ms,
            "recovery_evaluation_ms": total_recovery_evaluation_ms,
            "replan_ms": total_replan_ms,
            "execution_ms": total_execution_ms,
            "recovery_total_ms": recovery_trace.timing_ms.get("recovery_total_ms"),
        },
    }


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None, "max_ms": None, "n": 0}
    arr = np.array(values)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_g3.xml")
    parser.add_argument("--trigger", default="before_grasp", choices=VALID_TRIGGERS)
    parser.add_argument("--delta", type=float, nargs=3, default=[0.10, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20,
                         help="More than the usual 10 -- percentile estimates need it.")
    parser.add_argument("--normal-threshold", type=float, default=0.01)
    parser.add_argument("--recover-threshold", type=float, default=0.15)
    parser.add_argument("--out-json", default="results/g8_latency_budget.json")
    parser.add_argument("--out-md", default="results/LATENCY_BUDGET.md")
    args = parser.parse_args()

    evaluator = StateEvaluator(StateEvaluatorConfig(
        normal_threshold_m=args.normal_threshold, recover_threshold_m=args.recover_threshold,
    ))
    logger = EventLogger(run_id=f"G8-LATENCY-seed{args.seed}")

    trials = []
    for i in range(args.trials):
        trials.append(run_one_trial(args.model, args.trigger, tuple(args.delta), args.seed, i, evaluator, logger))
    logger.close()

    recovered = [t for t in trials if t["recovery"] is not None]
    n_no_recovery = len(trials) - len(recovered)

    components = {
        "perception_ms": [t["initial_perception_ms"] for t in trials],
        "evaluation_ms": [t["initial_evaluation_ms"] for t in trials],
        "observe_to_decision_ms": [t["observe_to_decision_ms"] for t in trials],
        "replan_ms": [t["recovery"]["replan_ms"] for t in recovered],
        "recovery_execution_ms": [t["recovery"]["execution_ms"] for t in recovered],
        "recovery_total_ms": [t["recovery"]["recovery_total_ms"] for t in recovered
                               if t["recovery"]["recovery_total_ms"] is not None],
    }

    summary = {name: percentiles(vals) for name, vals in components.items()}

    report = {
        "gate": "G8-H2",
        "config": {
            "model": args.model, "trigger": args.trigger, "delta": args.delta,
            "seed": args.seed, "n_trials": args.trials,
        },
        "n_trials_with_no_recovery_needed": n_no_recovery,
        "trials": trials,
        "summary_ms": summary,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    print("=== G8/H2 LATENCY SUMMARY ===")
    for name, s in summary.items():
        if s["n"] == 0:
            print(f"{name:24s}  (no data)")
            continue
        print(f"{name:24s} p50={s['p50_ms']:.3f}ms  p95={s['p95_ms']:.3f}ms  "
              f"mean={s['mean_ms']:.3f}ms  max={s['max_ms']:.3f}ms  (n={s['n']})")

    lines = [
        "# RoboGuardian — G8/H2 Latency Budget",
        "",
        f"Model: `{args.model}`  |  Seed: `{args.seed}`  |  Trials: `{args.trials}`  "
        f"({n_no_recovery} required no recovery)",
        "",
        "| Component | p50 (ms) | p95 (ms) | mean (ms) | max (ms) | n |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "perception_ms": "Perception (initial observe)",
        "evaluation_ms": "Evaluation (initial classify)",
        "observe_to_decision_ms": "Observe -> Decision",
        "replan_ms": "IK / Replan",
        "recovery_execution_ms": "Recovery execution (regrasp)",
        "recovery_total_ms": "Recovery total",
    }
    for name, s in summary.items():
        if s["n"] == 0:
            lines.append(f"| {labels[name]} | - | - | - | - | 0 |")
            continue
        lines.append(
            f"| {labels[name]} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | "
            f"{s['mean_ms']:.3f} | {s['max_ms']:.3f} | {s['n']} |"
        )
    lines += [
        "",
        "**What this measures**: wall-clock time inside this Python/MuJoCo process on "
        "this machine -- ground-truth `SimulationStateProvider`, not the OpenVINO vision "
        "path (see `results/g7_openvino.json` for perception-model inference latency "
        "separately). No speed claim is made here; the goal is to show the control loop's "
        "own overhead is small relative to the disturbance-recovery task, not to claim it "
        "is fast in an absolute sense.",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines))

    print(f"\nJSON: {out_json.resolve()}")
    print(f"Markdown: {Path(args.out_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
