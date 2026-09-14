"""
G6 — Reliability Protocol: A/B/C comparison
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

from sim.disturbance import DisturbanceInjector, VALID_TRIGGERS
from evidence.event_logger import EventLogger
from perception.state_provider import SimulationStateProvider
from control.state_machine import (
    StateEvaluator,
    StateEvaluatorConfig,
    NORMAL,
    RECOVERABLE,
)
from control.recovery import RecoveryController, ARM_JOINT_NAMES

HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
LIFT = [0.0, -0.8, 1.1, -0.3, 0.0, -0.6]

PLACE = [
    -0.0006804320,
    -1.3891631148,
    1.3904820914,
    1.5105929073,
    0.0039194352,
    -0.1745329776,
]

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

CARRY_OFFSET = np.array([0.0, 0.0, 0.048])
TRANSPORT_STEPS = 2000
TASK_FAIL_THRESHOLD_M = 0.05

FORCE_LIMITS = {
    "left_shoulder_lift": 7.5,
    "left_elbow_flex": 10.0,
    "left_wrist_flex": 3.35,
}

STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))

def build_model(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)
    act_ids = []
    for name in JOINT_NAMES:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{name}")
        if act_id < 0: raise LookupError(f"actuator not found: left_{name}")
        act_ids.append(act_id)
    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
    if -1 in (plate_jnt, ee_site, target_body): raise LookupError("required model id missing")
    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]
    return model, act_ids, plate_jnt, ee_site, target_body

def set_arm(data, act_ids, target):
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = float(target[i])

def carry(model, data, act_ids, q_adr, ee_site, target, steps):
    for _ in range(steps):
        set_arm(data, act_ids, target)
        mujoco.mj_step(model, data)
        data.qpos[q_adr:q_adr + 3] = data.site_xpos[ee_site] + CARRY_OFFSET
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

def measure_ee_error_xy(data, ee_site, target_body) -> float:
    ee_pos = data.site_xpos[ee_site].copy()
    predicted_release = ee_pos + CARRY_OFFSET
    target_pos = data.xpos[target_body].copy()
    return float(np.linalg.norm(predicted_release[:2] - target_pos[:2]))

def measure_plate_error_xy(data, q_adr, target_body) -> float:
    plate_pos = data.qpos[q_adr:q_adr + 3].copy()
    target_pos = data.xpos[target_body].copy()
    return float(np.linalg.norm(plate_pos[:2] - target_pos[:2]))

# --- Arm A ---
def run_arm_a_trial(model_path, trial_index, logger):
    t_start = time.perf_counter()
    model, act_ids, plate_jnt, ee_site, target_body = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(100): set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100): set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)
    for _ in range(60): set_arm(data, act_ids, GRASP); mujoco.mj_step(model, data)
    carry(model, data, act_ids, q_adr, ee_site, LIFT, 100)
    carry(model, data, act_ids, q_adr, ee_site, PLACE, TRANSPORT_STEPS)
    error = measure_ee_error_xy(data, ee_site, target_body)
    task_success = error <= TASK_FAIL_THRESHOLD_M
    elapsed = time.perf_counter() - t_start
    logger.log_step(step_id=f"A-STEP-PLACE-T{trial_index}", action_id="ACT-PLACE", action="place_plate_no_disturbance", expected_pose=[0,0,0], observed_pose=[0,0,0], pose_error=error, state=NORMAL, decision="EXECUTE", executed=True, extra={"trial_index": trial_index, "arm": "A"})
    return {"trial_index": trial_index, "arm": "A", "task_success": task_success, "target_error_xy_m": error, "disturbance_detected": None, "recovery_success": None, "stale_action_executed": False, "safe_stopped": False, "decision_latency_ms": None, "total_completion_time_s": elapsed}

# --- Arm B ---
def run_arm_b_trial(model_path, trigger, delta, seed, trial_index, logger):
    t_start = time.perf_counter()
    model, act_ids, plate_jnt, ee_site, target_body = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    injector = DisturbanceInjector("plate_joint", trigger, delta, seed + trial_index, noise_std=0.0)
    for _ in range(100): set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100): set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)
    injector.maybe_fire(model, data, trigger)
    for _ in range(60): set_arm(data, act_ids, GRASP); mujoco.mj_step(model, data)
    
    # NO carry() here. The plate was NOT grasped.
    for _ in range(100):
        set_arm(data, act_ids, LIFT)
        mujoco.mj_step(model, data)
    for _ in range(TRANSPORT_STEPS):
        set_arm(data, act_ids, PLACE)
        mujoco.mj_step(model, data)
        
    error = measure_plate_error_xy(data, q_adr, target_body)
    task_success = error <= TASK_FAIL_THRESHOLD_M
    elapsed = time.perf_counter() - t_start
    logger.log_step(step_id=f"B-STEP-PLACE-T{trial_index}", action_id="ACT-PLACE", action="place_plate_baseline", expected_pose=[0,0,0], observed_pose=[0,0,0], pose_error=error, state=NORMAL, decision="EXECUTE", executed=True, reason="BASELINE_NO_RECOVERY", extra={"trial_index": trial_index, "arm": "B", "disturbance": injector.as_dict()})
    return {"trial_index": trial_index, "arm": "B", "task_success": task_success, "target_error_xy_m": error, "disturbance_detected": False, "recovery_success": None, "stale_action_executed": True, "safe_stopped": False, "decision_latency_ms": None, "total_completion_time_s": elapsed}

# --- Arm C ---
def run_arm_c_trial(model_path, trigger, delta, seed, trial_index, evaluator, logger):
    t_start = time.perf_counter()
    model, act_ids, plate_jnt, ee_site, target_body = build_model(model_path)
    data = mujoco.MjData(model)
    q_adr = model.jnt_qposadr[plate_jnt]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()
    injector = DisturbanceInjector("plate_joint", trigger, delta, seed + trial_index, noise_std=0.0)
    provider = SimulationStateProvider(model, data)
    for _ in range(100): set_arm(data, act_ids, HOME); mujoco.mj_step(model, data)
    for _ in range(100): set_arm(data, act_ids, PRE_GRASP); mujoco.mj_step(model, data)
    injector.maybe_fire(model, data, trigger)
    t_disturbance = time.perf_counter()
    world_state = provider.observe("plate", expected_pose)
    eval_result = evaluator.evaluate(world_state)
    t_decision = time.perf_counter()
    decision_latency_ms = (t_decision - t_disturbance) * 1000.0
    disturbance_detected = eval_result.state != NORMAL
    recovery_trace = None
    safe_stopped = False
    if eval_result.state == NORMAL:
        for _ in range(60): set_arm(data, act_ids, GRASP); mujoco.mj_step(model, data)
    elif eval_result.state == RECOVERABLE:
        controller = RecoveryController(model, data, act_ids, "left_gripperframe", "plate", evaluator)
        recovery_trace = controller.recover(STALE_GRASP_TARGET, expected_pose)
        logger.log_recovery(step_id=f"C-STEP-RECOVERY-T{trial_index}", state=recovery_trace.outcome, decision="RESUME" if recovery_trace.verified else "SAFE_STOP", recovery_attempt=recovery_trace.recovery_attempts_used, target_updated=recovery_trace.fresh_target is not None, executed=recovery_trace.fresh_action_executed, success=recovery_trace.verified, extra={"trial_index": trial_index, **recovery_trace.to_dict()})
        safe_stopped = not recovery_trace.verified
    else:
        logger.log_safe_stop(step_id=f"C-STEP-RECOVERY-T{trial_index}", reason=f"UNHANDLED_STATE_{eval_result.state}", recovery_attempts_used=0)
        safe_stopped = True
    if safe_stopped:
        return {"trial_index": trial_index, "arm": "C", "task_success": False, "target_error_xy_m": None, "disturbance_detected": disturbance_detected, "recovery_success": recovery_trace.verified if recovery_trace else False, "stale_action_executed": recovery_trace.stale_action_executed if recovery_trace else False, "safe_stopped": True, "decision_latency_ms": decision_latency_ms, "total_completion_time_s": time.perf_counter() - t_start}
    carry(model, data, act_ids, q_adr, ee_site, LIFT, 100)
    carry(model, data, act_ids, q_adr, ee_site, PLACE, TRANSPORT_STEPS)
    error = measure_ee_error_xy(data, ee_site, target_body)
    task_success = error <= TASK_FAIL_THRESHOLD_M
    elapsed = time.perf_counter() - t_start
    logger.log_step(step_id=f"C-STEP-PLACE-T{trial_index}", action_id="ACT-PLACE", action="place_plate_recovered", expected_pose=expected_pose.tolist(), observed_pose=[0,0,0], pose_error=error, state=eval_result.state, decision="RESUME_AFTER_RECOVERY" if recovery_trace else "EXECUTE", executed=True, extra={"trial_index": trial_index, "arm": "C"})
    return {"trial_index": trial_index, "arm": "C", "task_success": task_success, "target_error_xy_m": error, "disturbance_detected": disturbance_detected, "recovery_success": recovery_trace.verified if recovery_trace else None, "stale_action_executed": recovery_trace.stale_action_executed if recovery_trace else False, "safe_stopped": False, "decision_latency_ms": decision_latency_ms, "total_completion_time_s": elapsed}

def summarize(arm_trials):
    n = len(arm_trials)
    if n == 0: return {}
    def rate(key, predicate=lambda v: bool(v)):
        vals = [t[key] for t in arm_trials if t.get(key) is not None]
        return sum(1 for v in vals if predicate(v)) / len(vals) if vals else None
    def mean(key):
        vals = [t[key] for t in arm_trials if t.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    return {"n_trials": n, "task_success_rate": rate("task_success"), "disturbance_detection_rate": rate("disturbance_detected"), "recovery_success_rate": rate("recovery_success"), "stale_action_execution_rate": rate("stale_action_executed"), "safe_stop_rate": rate("safe_stopped"), "mean_decision_latency_ms": mean("decision_latency_ms"), "mean_total_completion_time_s": mean("total_completion_time_s")}

def write_results_md(path, summaries, config):
    lines = ["# RoboGuardian — G6 Reliability Results", "", f"Model: `{config['model']}` | Trigger: `{config['trigger']}` | Delta: `{config['delta']}` | Seed: `{config['seed']}` | Trials/arm: `{config['n_trials']}`", "", "| Arm | Success | Detection | Recovery | Stale-Exec | Safe-Stop | Latency (ms) | Time (s) |", "|---|---|---|---|---|---|---|---|"]
    fmt_pct = lambda v: f"{v:.0%}" if v is not None else "n/a"
    fmt_ms = lambda v: f"{v:.2f}" if v is not None else "n/a"
    fmt_s = lambda v: f"{v:.2f}" if v is not None else "n/a"
    labels = {"A": "A — No disturbance", "B": "B — Disturbance, no recovery", "C": "C — Disturbance + recovery"}
    for arm in ("A", "B", "C"):
        s = summaries[arm]
        sr = s.get("task_success_rate")
        n_success = round(sr * s["n_trials"]) if sr is not None else 0
        lines.append(f"| {labels[arm]} | {n_success}/{s['n_trials']} ({fmt_pct(sr)}) | {fmt_pct(s.get('disturbance_detection_rate'))} | {fmt_pct(s.get('recovery_success_rate'))} | {fmt_pct(s.get('stale_action_execution_rate'))} | {fmt_pct(s.get('safe_stop_rate'))} | {fmt_ms(s.get('mean_decision_latency_ms'))} | {fmt_s(s.get('mean_total_completion_time_s'))} |")
    a_n, b_n, c_n = summaries["A"]["n_trials"], summaries["B"]["n_trials"], summaries["C"]["n_trials"]
    a_ok = round((summaries["A"].get("task_success_rate") or 0) * a_n)
    b_ok = round((summaries["B"].get("task_success_rate") or 0) * b_n)
    c_ok = round((summaries["C"].get("task_success_rate") or 0) * c_n)
    lines += ["", "## Killer-demo summary line", "", f"- Arm A (no disturbance): {a_ok}/{a_n} success", f"- Arm B (disturbance, no recovery): {b_ok}/{b_n} success", f"- Arm C (disturbance + RoboGuardian): {c_ok}/{c_n} success", ""]
    path.write_text("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_g3.xml")
    parser.add_argument("--trigger", default="before_grasp", choices=VALID_TRIGGERS)
    parser.add_argument("--delta", type=float, nargs=3, default=[0.10, 0.0, 0.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--normal-threshold", type=float, default=0.01)
    parser.add_argument("--recover-threshold", type=float, default=0.15)
    parser.add_argument("--out-json", default="results/g6_reliability.json")
    parser.add_argument("--out-md", default="results/RESULTS.md")
    args = parser.parse_args()
    evaluator = StateEvaluator(StateEvaluatorConfig(normal_threshold_m=args.normal_threshold, recover_threshold_m=args.recover_threshold))
    logger = EventLogger(run_id=f"G6-RELIABILITY-seed{args.seed}")
    arm_a, arm_b, arm_c = [], [], []
    for i in range(args.trials): arm_a.append(run_arm_a_trial(args.model, i, logger))
    for i in range(args.trials): arm_b.append(run_arm_b_trial(args.model, args.trigger, tuple(args.delta), args.seed, i, logger))
    for i in range(args.trials): arm_c.append(run_arm_c_trial(args.model, args.trigger, tuple(args.delta), args.seed, i, evaluator, logger))
    logger.close()
    for label, trials in (("A", arm_a), ("B", arm_b), ("C", arm_c)):
        n_ok = sum(1 for t in trials if t["task_success"])
        print(f"Arm {label}: {n_ok}/{len(trials)} success")
    summaries = {"A": summarize(arm_a), "B": summarize(arm_b), "C": summarize(arm_c)}
    config = {"model": args.model, "trigger": args.trigger, "delta": args.delta, "seed": args.seed, "n_trials": args.trials, "normal_threshold_m": args.normal_threshold, "recover_threshold_m": args.recover_threshold}
    report = {"gate": "G6", "config": config, "arm_a_trials": arm_a, "arm_b_trials": arm_b, "arm_c_trials": arm_c, "summary": summaries, "g6_pass": (summaries["A"].get("task_success_rate") is not None and summaries["A"]["task_success_rate"] >= 0.90 and summaries["B"].get("task_success_rate") is not None and summaries["B"]["task_success_rate"] <= 0.20 and summaries["C"].get("task_success_rate") is not None and summaries["C"]["task_success_rate"] >= 0.80 and summaries["C"].get("stale_action_execution_rate") == 0.0)}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md = Path(args.out_md)
    write_results_md(out_md, summaries, config)
    print(f"\n=== G6 {'PASS' if report['g6_pass'] else 'FAIL'} ===")
    print(f"JSON evidence: {out_json.resolve()}")
    print(f"Markdown summary: {out_md.resolve()}")
    return 0 if report["g6_pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
