"""
Recovery Controller (Phase E / G5)
=====================================

Minimum recovery loop from build plan section 10:

    RECOVERABLE
      -> pause stale action
      -> re-observe target
      -> update target pose
      -> regrasp
      -> verify result
      -> resume

Hard rule from section 18: recovery must use FRESH observation only.
This controller structurally cannot execute the stale target -- recover()
takes it only as a value to log for evidence (`trace.stale_target`), and
nothing in the method body ever writes it to ctrl. The only joint values
ever sent to the actuators come from a fresh solve_arm_ik() call seeded
by a fresh provider.observe().

Bounded by MAX_RECOVERY_ATTEMPTS / MAX_REOBSERVATION_ATTEMPTS (section
12): exhausting the budget returns SAFE_STOP, never an unbounded retry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from control.ik_solver import solve_arm_ik
from control.state_machine import StateEvaluator, NORMAL, SAFE_STOP, UNCERTAIN
from perception.state_provider import SimulationStateProvider

MAX_RECOVERY_ATTEMPTS = 2
MAX_REOBSERVATION_ATTEMPTS = 2

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_CLOSE_VALUE = -0.6
GRASP_VERIFY_THRESHOLD_M = 0.03


@dataclass
class RecoveryTrace:
    stale_target: dict
    fresh_target: dict | None = None
    stale_action_executed: bool = False   # structurally always False -- see module docstring
    fresh_action_executed: bool = False
    reobservations_used: int = 0
    recovery_attempts_used: int = 0
    verified: bool = False
    outcome: str = SAFE_STOP
    attempts_log: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)  # H2: end-to-end critical-path timing

    def to_dict(self) -> dict:
        return {
            "stale_target": self.stale_target,
            "fresh_target": self.fresh_target,
            "stale_action_executed": self.stale_action_executed,
            "fresh_action_executed": self.fresh_action_executed,
            "reobservations_used": self.reobservations_used,
            "recovery_attempts_used": self.recovery_attempts_used,
            "verified": self.verified,
            "outcome": self.outcome,
            "attempts_log": self.attempts_log,
            "timing_ms": self.timing_ms,
        }


class RecoveryController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        act_ids: list[int],
        ee_site_name: str,
        object_body_name: str,
        evaluator: StateEvaluator,
        provider=None,
    ):
        self.model = model
        self.data = data
        self.act_ids = act_ids
        self.ee_site_name = ee_site_name
        self.object_body_name = object_body_name
        self.evaluator = evaluator
        # Defaults to ground-truth simulation observation (G4/G5 behavior).
        # Pass a VisionStateProvider here to run the exact same recovery
        # logic against real (OpenVINO) perception instead -- nothing else
        # in this class needs to change.
        self.provider = provider if provider is not None else SimulationStateProvider(model, data)

    def _execute_arm(self, target_values: dict, gripper_value: float, steps: int) -> None:
        for _ in range(steps):
            for i, name in enumerate(ARM_JOINT_NAMES):
                self.data.ctrl[self.act_ids[i]] = float(target_values[name])
            self.data.ctrl[self.act_ids[len(ARM_JOINT_NAMES)]] = float(gripper_value)
            mujoco.mj_step(self.model, self.data)

    def recover(self, stale_target: dict, expected_pose: np.ndarray) -> RecoveryTrace:
        """stale_target is logged for evidence only -- see module docstring
        for why it structurally can never reach the actuators from here.
        expected_pose is the pre-disturbance pose the invalidated plan was
        based on; also logged only, never re-used as an observation."""
        trace = RecoveryTrace(stale_target=dict(stale_target))
        t_recover_start = time.perf_counter()

        for attempt in range(1, MAX_RECOVERY_ATTEMPTS + 1):
            trace.recovery_attempts_used = attempt
            attempt_log = {"attempt": attempt}
            attempt_timing = {}
            t_attempt_start = time.perf_counter()

            fresh_state = None
            eval_result = None
            observation_ms_total = 0.0
            evaluation_ms_total = 0.0
            for _reobs in range(1, MAX_REOBSERVATION_ATTEMPTS + 1):
                trace.reobservations_used += 1

                t0 = time.perf_counter()
                fresh_state = self.provider.observe(self.object_body_name, expected_pose)
                t1 = time.perf_counter()
                eval_result = self.evaluator.evaluate(fresh_state)
                t2 = time.perf_counter()

                observation_ms_total += (t1 - t0) * 1000.0
                evaluation_ms_total += (t2 - t1) * 1000.0

                if eval_result.state != UNCERTAIN:
                    break
            attempt_log["observed_pose"] = fresh_state.observed_pose.tolist()
            attempt_log["evaluated_state"] = eval_result.state if eval_result else None
            attempt_timing["observation_ms"] = observation_ms_total
            attempt_timing["evaluation_ms"] = evaluation_ms_total

            if eval_result is None or eval_result.state == UNCERTAIN:
                attempt_log["result"] = "REOBSERVATION_BUDGET_EXHAUSTED"
                attempt_timing["attempt_total_ms"] = (time.perf_counter() - t_attempt_start) * 1000.0
                attempt_log["timing_ms"] = attempt_timing
                trace.attempts_log.append(attempt_log)
                trace.outcome = SAFE_STOP
                trace.timing_ms["recovery_total_ms"] = (time.perf_counter() - t_recover_start) * 1000.0
                return trace

            t_replan_start = time.perf_counter()
            ik_result = solve_arm_ik(
                model=self.model,
                site_name=self.ee_site_name,
                joint_names=ARM_JOINT_NAMES,
                goal_pos=fresh_state.observed_pose,
                seed_qpos={n: v for n, v in stale_target.items() if n in ARM_JOINT_NAMES},
            )
            t_replan_end = time.perf_counter()
            replan_ms = (t_replan_end - t_replan_start) * 1000.0
            attempt_timing["replan_ms"] = replan_ms
            # Observe -> evaluate -> replan, all BEFORE any actuator command is
            # issued -- this is the "decision to fresh action" latency H2 asks for.
            attempt_timing["decision_to_fresh_action_ms"] = (
                observation_ms_total + evaluation_ms_total + replan_ms
            )

            attempt_log["ik_converged"] = ik_result["converged"]
            attempt_log["ik_error_m"] = ik_result["final_position_error_m"]

            if not ik_result["converged"]:
                attempt_log["result"] = "IK_DID_NOT_CONVERGE"
                attempt_timing["attempt_total_ms"] = (time.perf_counter() - t_attempt_start) * 1000.0
                attempt_log["timing_ms"] = attempt_timing
                trace.attempts_log.append(attempt_log)
                continue  # retry within budget rather than execute an unreachable target

            trace.fresh_target = ik_result["joint_solution_rad"]

            t_exec_start = time.perf_counter()
            self._execute_arm(trace.fresh_target, GRIPPER_CLOSE_VALUE, steps=60)
            t_exec_end = time.perf_counter()
            attempt_timing["execution_ms"] = (t_exec_end - t_exec_start) * 1000.0
            trace.fresh_action_executed = True

            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
            ee_pos = self.data.site_xpos[ee_site_id].copy()
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name)
            obj_pos = self.data.xpos[body_id].copy()
            grasp_error = float(np.linalg.norm(ee_pos - obj_pos))
            attempt_log["grasp_error_m"] = grasp_error

            attempt_timing["attempt_total_ms"] = (time.perf_counter() - t_attempt_start) * 1000.0
            attempt_log["timing_ms"] = attempt_timing

            if grasp_error <= GRASP_VERIFY_THRESHOLD_M:
                attempt_log["result"] = "VERIFIED"
                trace.attempts_log.append(attempt_log)
                trace.verified = True
                trace.outcome = NORMAL
                trace.timing_ms["recovery_total_ms"] = (time.perf_counter() - t_recover_start) * 1000.0
                return trace

            attempt_log["result"] = "GRASP_VERIFY_FAILED"
            trace.attempts_log.append(attempt_log)
            # loop again: re-observe + retry within remaining recovery budget

        trace.outcome = SAFE_STOP
        trace.timing_ms["recovery_total_ms"] = (time.perf_counter() - t_recover_start) * 1000.0
        return trace
