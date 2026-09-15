"""
FaultInjectingProvider (G8 / H1 robustness matrix)
======================================================

Wraps any StateProvider (SimulationStateProvider or VisionStateProvider)
to inject controlled degradations on top of it: measurement noise,
observation delay, and dropped frames. Implements the same
observe(object_name, expected_pose) -> WorldState interface, so
control/state_machine.py and control/recovery.py consume it with zero
changes -- this is deliberate, per the hard rule "ห้ามเปลี่ยน core
recovery architecture".

Semantics (each is opt-in via its own parameter, all default to "off"
so wrapping a provider with no fault args is a no-op passthrough):

- noise_std: Gaussian noise added to observed_pose, seeded for
  reproducibility. pose_error is recomputed against the noisy pose.

- delay_frames: this control loop only observes at discrete trigger
  boundaries, not a fixed-frequency polling loop -- so "delay" here
  means "N observe() calls behind", not wall-clock ms. The table's
  50ms/100ms labels are a modeling proxy mapped to delay_frames=1/2;
  state that explicitly wherever this class's results are reported.

- drop_frame_count: the first N observe() calls in this provider's
  lifetime return the previous call's result again (a stale duplicate,
  confidence halved) instead of a fresh reading -- deterministic count,
  not a probability, matching the build plan's "0/1/2 frames" framing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from perception.state_provider import WorldState


class FaultInjectingProvider:
    def __init__(
        self,
        inner_provider,
        noise_std: float = 0.0,
        delay_frames: int = 0,
        drop_frame_count: int = 0,
        seed: int = 0,
    ):
        self._inner = inner_provider
        self._noise_std = noise_std
        self._delay_frames = delay_frames
        self._drop_frame_count = drop_frame_count
        self._rng = np.random.default_rng(seed)

        self._history: deque[WorldState] = deque(maxlen=delay_frames + 1)
        self._last_delivered: WorldState | None = None
        self._call_index = 0

    def _apply_noise(self, state: WorldState) -> WorldState:
        if self._noise_std <= 0:
            return state
        noisy_pose = state.observed_pose + self._rng.normal(0.0, self._noise_std, size=3)
        pose_error = float(np.linalg.norm(noisy_pose - state.expected_pose))
        extra = dict(state.extra)
        extra["fault_noise_std"] = self._noise_std
        return replace(state, observed_pose=noisy_pose, pose_error=pose_error, extra=extra)

    def observe(self, object_name: str, expected_pose) -> WorldState:
        self._call_index += 1

        fresh = self._apply_noise(self._inner.observe(object_name, expected_pose))
        self._history.append(fresh)
        delayed = self._history[0]  # oldest buffered entry -> the "N frames behind" reading

        if self._call_index <= self._drop_frame_count:
            if self._last_delivered is not None:
                extra = dict(self._last_delivered.extra)
                extra["fault_dropped_frame"] = True
                delivered = replace(
                    self._last_delivered,
                    confidence=self._last_delivered.confidence * 0.5,
                    extra=extra,
                )
            else:
                delivered = delayed  # nothing to fall back to on the very first call
        else:
            delivered = delayed

        self._last_delivered = delivered
        return delivered

    def close(self) -> None:
        if hasattr(self._inner, "close"):
            self._inner.close()
