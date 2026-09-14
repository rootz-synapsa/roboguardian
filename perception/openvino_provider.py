"""
VisionStateProvider (Phase G / G7)
=====================================

Same interface as SimulationStateProvider: observe(object_name,
expected_pose) -> WorldState. Nothing in control/state_machine.py or
control/recovery.py needs to change to consume this instead -- that
boundary is the entire point of the StateProvider abstraction from
section 9.

Renders the current MuJoCo scene, runs it through the OpenVINO-compiled
plate-position regressor, and de-normalizes the output back into world
coordinates using the stats saved by train_detector.py.

confidence here is a placeholder constant (1.0) -- this model has no
calibrated confidence head. Treat any UNCERTAIN-state behavior that
depends on confidence as not yet meaningful until that's added; don't
claim otherwise in the submission writeup (section 35).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import openvino as ov
from PIL import Image

from perception.state_provider import WorldState


class VisionStateProvider:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        onnx_model_path: str,
        norm_stats_path: str,
        device: str = "CPU",
    ):
        self._model = model
        self._data = data

        norm = json.loads(Path(norm_stats_path).read_text())
        self._x_mean, self._x_std = norm["x_mean"], norm["x_std"]
        self._y_mean, self._y_std = norm["y_mean"], norm["y_std"]
        self._z_fixed = norm["z_fixed"]
        self._img_size = norm["img_size"]

        # Same camera/render-size/crop pipeline as dataset_gen.py -- read
        # from the norm file rather than re-specified here, so training
        # and inference can never silently drift apart.
        self._camera = norm["camera"]
        self._render_size = norm["render_size"]
        self._crop_rows = tuple(norm["crop_rows"])
        self._crop_cols = tuple(norm["crop_cols"])

        self._renderer = mujoco.Renderer(model, height=self._render_size, width=self._render_size)

        core = ov.Core()
        compiled = core.compile_model(onnx_model_path, device)
        self._infer_request = compiled.create_infer_request()

    def observe(self, object_name: str, expected_pose: np.ndarray) -> WorldState:
        self._renderer.update_scene(self._data, camera=self._camera)
        raw_frame = self._renderer.render()  # HxWx3 uint8 at render_size
        cropped = raw_frame[self._crop_rows[0]:self._crop_rows[1],
                             self._crop_cols[0]:self._crop_cols[1]]
        resized = np.array(Image.fromarray(cropped).resize(
            (self._img_size, self._img_size), Image.BILINEAR))

        frame = resized.astype(np.float32) / 255.0
        img = np.transpose(frame, (2, 0, 1))[None, ...]  # 1x3xHxW

        output = self._infer_request.infer({0: img})
        xy_norm = list(output.values())[0][0]

        x = float(xy_norm[0]) * self._x_std + self._x_mean
        y = float(xy_norm[1]) * self._y_std + self._y_mean
        observed = np.array([x, y, self._z_fixed])

        expected = np.asarray(expected_pose, dtype=float)
        pose_error = float(np.linalg.norm(observed - expected))

        return WorldState(
            object_name=object_name,
            visible=True,
            expected_pose=expected,
            observed_pose=observed,
            pose_error=pose_error,
            confidence=1.0,
            extra={"source": "openvino_vision"},
        )

    def close(self) -> None:
        self._renderer.close()
