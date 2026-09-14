"""
Controlled Disturbance Injector
================================

Deterministic, seedable disturbance mechanism for Phase C of the build
plan. Moves a named object (via its freejoint qpos) by a fixed delta the
first time a named trigger point fires during a run.

Design intent (per build plan section 16):
  - deterministic: same seed + same trigger -> same displacement, every time
  - fires exactly once per run, at a named boundary -- not continuously
  - object identity, trigger point, and delta are all explicit, logged
    inputs, never hidden inside the control loop

Usage:
    injector = DisturbanceInjector(
        object_joint_name="plate_joint",
        trigger="before_grasp",
        delta=(0.10, 0.0, 0.0),
        seed=42,
    )

    # in your trajectory loop, right before entering the GRASP phase:
    injector.maybe_fire(model, data, current_trigger="before_grasp")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

VALID_TRIGGERS = (
    "before_approach",
    "before_grasp",
    "after_grasp",
    "before_transport",
    "before_placement",
    "after_placement",
)


@dataclass
class DisturbanceEvent:
    object_joint_name: str
    trigger: str
    delta: tuple[float, float, float]
    seed: int
    fired: bool = False
    pose_before: list[float] | None = None
    pose_after: list[float] | None = None


class DisturbanceInjector:
    def __init__(
        self,
        object_joint_name: str,
        trigger: str,
        delta: tuple[float, float, float],
        seed: int,
        noise_std: float = 0.0,
    ):
        if trigger not in VALID_TRIGGERS:
            raise ValueError(f"unknown trigger '{trigger}', expected one of {VALID_TRIGGERS}")

        self.object_joint_name = object_joint_name
        self.trigger = trigger
        self.base_delta = np.array(delta, dtype=float)
        self.seed = seed
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        self.event = DisturbanceEvent(
            object_joint_name=object_joint_name,
            trigger=trigger,
            delta=delta,
            seed=seed,
        )

    def maybe_fire(self, model: mujoco.MjModel, data: mujoco.MjData, current_trigger: str) -> bool:
        """Call this at each named boundary in your trajectory loop.
        Fires (mutates qpos) exactly once, the first time current_trigger
        matches the configured trigger. Returns True iff it fired this call."""
        if self.event.fired or current_trigger != self.trigger:
            return False

        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)
        if jnt_id < 0:
            raise LookupError(f"joint not found: {self.object_joint_name}")

        q_adr = model.jnt_qposadr[jnt_id]
        before = data.qpos[q_adr:q_adr + 3].copy()

        # noise_std=0.0 by default -> fully deterministic delta, matching
        # the "same disturbance every trial" requirement in section 16.
        # A nonzero noise_std is opt-in, for later robustness testing only.
        jitter = self._rng.normal(0.0, self.noise_std, size=3) if self.noise_std > 0 else 0.0
        applied_delta = self.base_delta + jitter

        data.qpos[q_adr:q_adr + 3] = before + applied_delta
        mujoco.mj_forward(model, data)

        after = data.qpos[q_adr:q_adr + 3].copy()

        self.event.fired = True
        self.event.pose_before = before.tolist()
        self.event.pose_after = after.tolist()
        return True

    def as_dict(self) -> dict:
        return {
            "object": self.event.object_joint_name,
            "trigger": self.event.trigger,
            "delta": list(self.event.delta),
            "seed": self.event.seed,
            "fired": self.event.fired,
            "pose_before": self.event.pose_before,
            "pose_after": self.event.pose_after,
        }
