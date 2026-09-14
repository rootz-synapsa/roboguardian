"""
Simulation State Provider — G4 detection layer.

Reads the current pose of a named object from the MuJoCo simulation
and compares it against the expected (planned) pose. Returns a
WorldState that the StateEvaluator can classify.

This is the "observe" half of the observe -> evaluate -> decide loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass
class WorldState:
    """Snapshot of one object's observed vs expected state."""

    object_name: str
    expected_pose: np.ndarray
    observed_pose: np.ndarray
    pose_error: float
    # Optional fields for vision-based providers (G7)
    visible: bool = True
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


class SimulationStateProvider:
    """Reads object poses directly from MuJoCo sim state.

    In a real robot this would be replaced by a perception pipeline
    (camera -> OpenVINO -> pose estimation). The interface stays the same.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def observe(
        self,
        object_name: str,
        expected_pose: np.ndarray,
    ) -> WorldState:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_name,
        )
        if body_id < 0:
            raise LookupError(f"body not found: {object_name}")

        observed = self.data.xpos[body_id].copy()
        expected = np.asarray(expected_pose, dtype=float)
        error = float(np.linalg.norm(observed - expected))

        return WorldState(
            object_name=object_name,
            expected_pose=expected,
            observed_pose=observed,
            pose_error=error,
        )
