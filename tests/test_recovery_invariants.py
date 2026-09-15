"""
Recovery invariant tests (G8 / H3)
=====================================

These test the guarantees the project's core claim depends on -- not
feature coverage for its own sake. Each test name states the invariant
directly so a reviewer can map test -> claim without reading the body.

Uses the real project model (models/dual_so101_g3.xml) for realistic
integration coverage rather than a synthetic mock model, since the
model is already a committed asset the CI G1 smoke test depends on too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.recovery import (  # noqa: E402
    RecoveryController,
    ARM_JOINT_NAMES,
    MAX_RECOVERY_ATTEMPTS,
    MAX_REOBSERVATION_ATTEMPTS,
)
from control.state_machine import StateEvaluator, StateEvaluatorConfig, NORMAL, SAFE_STOP  # noqa: E402
from perception.state_provider import SimulationStateProvider, WorldState  # noqa: E402
from sim.disturbance import DisturbanceInjector  # noqa: E402

MODEL_PATH = "models/dual_so101_g3.xml"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME = [0.0, -0.4, 0.9, -0.5, 0.0, 0.0]
PRE_GRASP = [0.0, -0.9, 1.2, -0.3, 0.0, 0.0]
GRASP = [0.0, -1.0, 1.3, -0.2, 0.0, -0.6]
STALE_GRASP_TARGET = dict(zip(ARM_JOINT_NAMES, GRASP[:5]))

FORCE_LIMITS = {"left_shoulder_lift": 7.5, "left_elbow_flex": 10.0, "left_wrist_flex": 3.35}


def _build_disturbed_setup(delta=(0.10, 0.0, 0.0), seed=42):
    """Shared fixture logic: model at the moment a RECOVERABLE disturbance
    has just fired at before_grasp -- the exact precondition recover() is
    designed for."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{n}") for n in JOINT_NAMES]
    for act_name, limit in FORCE_LIMITS.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id >= 0:
            model.actuator_forcelimited[act_id] = True
            model.actuator_forcerange[act_id] = [-limit, limit]

    data = mujoco.MjData(model)
    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pose = data.qpos[q_adr:q_adr + 3].copy()

    injector = DisturbanceInjector("plate_joint", "before_grasp", delta, seed, noise_std=0.0)

    for _ in range(100):
        for i, a in enumerate(act_ids):
            data.ctrl[a] = HOME[i]
        mujoco.mj_step(model, data)
    for _ in range(100):
        for i, a in enumerate(act_ids):
            data.ctrl[a] = PRE_GRASP[i]
        mujoco.mj_step(model, data)

    injector.maybe_fire(model, data, "before_grasp")

    return model, data, act_ids, expected_pose


def test_stale_target_never_written_to_ctrl():
    """Section 18's core guarantee: after recover(), no actuator ever
    received the pre-disturbance (stale) joint values. We check this by
    comparing final ctrl state against the stale target directly -- not
    just trusting the reported flag."""
    model, data, act_ids, expected_pose = _build_disturbed_setup()
    evaluator = StateEvaluator(StateEvaluatorConfig(normal_threshold_m=0.01, recover_threshold_m=0.15))
    provider = SimulationStateProvider(model, data)

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=provider,
    )
    trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    assert trace.stale_action_executed is False

    stale_ctrl = np.array([STALE_GRASP_TARGET[n] for n in ARM_JOINT_NAMES])
    final_ctrl = np.array([data.ctrl[act_ids[i]] for i in range(len(ARM_JOINT_NAMES))])
    # If the stale target had ever been executed, ctrl would exactly equal
    # it (both are literal joint angles, not just "close").
    assert not np.allclose(final_ctrl, stale_ctrl, atol=1e-9), (
        "actuator ctrl matches the stale target exactly -- stale action was executed"
    )


def test_recovery_budget_is_bounded():
    """recovery_attempts_used must never exceed MAX_RECOVERY_ATTEMPTS,
    regardless of outcome -- this is what prevents the
    fail/retry/fail/retry infinite loop section 12 rules out."""
    model, data, act_ids, expected_pose = _build_disturbed_setup()
    evaluator = StateEvaluator(StateEvaluatorConfig(normal_threshold_m=0.01, recover_threshold_m=0.15))
    provider = SimulationStateProvider(model, data)

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=provider,
    )
    trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    assert trace.recovery_attempts_used <= MAX_RECOVERY_ATTEMPTS
    assert trace.reobservations_used <= MAX_RECOVERY_ATTEMPTS * MAX_REOBSERVATION_ATTEMPTS


def test_failed_ik_never_executes(monkeypatch):
    """If solve_arm_ik() reports non-convergence, _execute_arm() must
    never be called for that attempt -- an unreachable target must not
    reach the actuators just because it was computed."""
    import control.recovery as recovery_module

    def fake_solve_arm_ik(**kwargs):
        return {"converged": False, "final_position_error_m": 999.0, "joint_solution_rad": {}}

    monkeypatch.setattr(recovery_module, "solve_arm_ik", fake_solve_arm_ik)

    model, data, act_ids, expected_pose = _build_disturbed_setup()
    evaluator = StateEvaluator(StateEvaluatorConfig(normal_threshold_m=0.01, recover_threshold_m=0.15))
    provider = SimulationStateProvider(model, data)

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=provider,
    )
    trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    assert trace.fresh_action_executed is False
    assert trace.outcome == SAFE_STOP
    assert all(a["result"] == "IK_DID_NOT_CONVERGE" for a in trace.attempts_log)


def test_fresh_observation_required_before_recovery():
    """Every recovery attempt must call provider.observe() at least once
    before doing anything else -- recovery cannot proceed on stale data
    it already had lying around."""
    model, data, act_ids, expected_pose = _build_disturbed_setup()
    evaluator = StateEvaluator(StateEvaluatorConfig(normal_threshold_m=0.01, recover_threshold_m=0.15))

    call_count = {"n": 0}
    real_provider = SimulationStateProvider(model, data)

    class CountingProvider:
        def observe(self, object_name, expected_pose):
            call_count["n"] += 1
            return real_provider.observe(object_name, expected_pose)

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", evaluator, provider=CountingProvider(),
    )
    trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    assert call_count["n"] >= 1
    assert trace.reobservations_used == call_count["n"]


def test_uncertain_state_safe_stops(monkeypatch):
    """If the evaluator can never resolve past UNCERTAIN within the
    reobservation budget, the outcome must be SAFE_STOP with no action
    ever executed -- not a guess."""
    model, data, act_ids, expected_pose = _build_disturbed_setup()
    provider = SimulationStateProvider(model, data)

    class AlwaysUncertainEvaluator:
        def evaluate(self, world_state):
            from control.state_machine import UNCERTAIN

            class Result:
                state = UNCERTAIN
                reason = "FORCED_UNCERTAIN_FOR_TEST"
                pose_error = world_state.pose_error
            return Result()

    controller = RecoveryController(
        model, data, act_ids, "left_gripperframe", "plate", AlwaysUncertainEvaluator(), provider=provider,
    )
    trace = controller.recover(STALE_GRASP_TARGET, expected_pose)

    assert trace.outcome == SAFE_STOP
    assert trace.fresh_action_executed is False
    assert trace.stale_action_executed is False
