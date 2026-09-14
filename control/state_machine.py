"""
State Evaluator — G4 detection logic.

Classifies a WorldState into one of the governance states:
  NORMAL        — object is where expected, proceed
  RECOVERABLE   — object displaced but recoverable, pause & re-plan
  UNRECOVERABLE — object displaced beyond recovery, safe stop
  SAFE_STOP     — terminal state, do not execute

Thresholds are configurable via StateEvaluatorConfig.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- State constants ---------------------------------------------------
NORMAL = "NORMAL"
RECOVERABLE = "RECOVERABLE"
UNRECOVERABLE = "UNRECOVERABLE"
UNCERTAIN = "UNCERTAIN"
SAFE_STOP = "SAFE_STOP"


@dataclass
class StateEvaluatorConfig:
    normal_threshold_m: float = 0.01
    recover_threshold_m: float = 0.15


@dataclass
class EvalResult:
    state: str
    reason: str
    pose_error: float


class StateEvaluator:
    """Pure function: WorldState → EvalResult.

    No side effects, no logging, no action — just classification.
    The caller decides what to do with the result.
    """

    def __init__(self, config: StateEvaluatorConfig):
        self.config = config

    def evaluate(self, world_state) -> EvalResult:
        error = world_state.pose_error

        if error <= self.config.normal_threshold_m:
            return EvalResult(
                state=NORMAL,
                reason="WITHIN_NORMAL_THRESHOLD",
                pose_error=error,
            )

        if error <= self.config.recover_threshold_m:
            return EvalResult(
                state=RECOVERABLE,
                reason="OBJECT_DISPLACED_RECOVERABLE",
                pose_error=error,
            )

        return EvalResult(
            state=UNRECOVERABLE,
            reason="OBJECT_DISPLACED_UNRECOVERABLE",
            pose_error=error,
        )
