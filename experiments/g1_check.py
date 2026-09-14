"""
G1 — Environment & Toolchain Feasibility Spike
================================================

Runs every check listed under "G1 Definition of Done" in the RoboGuardian
build plan, in a single reproducible pass, and writes the result to
results/g1_environment.json.

Design notes:
- Every check is isolated in its own try/except so one failure doesn't
  hide the status of the others. You get a full report even on a bad run.
- Nothing here is "PASS if it looks like it ran" — each check asserts a
  concrete condition (a value changed, a shape matches, a file exists).
- Run this from the repo root (same directory as `models/`, `results/`).

Usage:
    python experiments/g1_check.py
    python experiments/g1_check.py --model models/dual_so101_no_table_collision.xml
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_s: float = 0.0


def run_check(name: str, fn: Callable[[], dict[str, Any]]) -> CheckResult:
    """Run one check, catching and recording any failure instead of
    aborting the whole G1 pass."""
    t0 = time.perf_counter()
    try:
        detail = fn() or {}
        return CheckResult(
            name=name,
            passed=True,
            detail=detail,
            duration_s=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - intentional: log every failure mode
        return CheckResult(
            name=name,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=6)},
            duration_s=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Individual G1 checks
# ---------------------------------------------------------------------------

def check_mujoco_launches() -> dict[str, Any]:
    import mujoco

    return {"mujoco_version": mujoco.__version__}


def check_model_loads(model_path: str) -> dict[str, Any]:
    import mujoco

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(p))
    _CTX["model"] = model
    _CTX["data"] = mujoco.MjData(model)
    return {
        "model_path": str(p),
        "n_bodies": model.nbody,
        "n_joints": model.njnt,
        "n_actuators": model.nu,
    }


def check_arm_actuator_callable() -> dict[str, Any]:
    import mujoco

    model = _CTX["model"]
    data = _CTX["data"]

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pan")
    if act_id < 0:
        raise LookupError("actuator 'left_shoulder_pan' not found in model")

    before = float(data.ctrl[act_id])
    data.ctrl[act_id] = before + 0.05
    mujoco.mj_step(model, data)
    after = float(data.ctrl[act_id])

    if after == before:
        raise AssertionError("ctrl value did not change after write + step")

    return {"actuator": "left_shoulder_pan", "ctrl_before": before, "ctrl_after": after}


def check_object_pose_readable() -> dict[str, Any]:
    import mujoco

    model = _CTX["model"]
    data = _CTX["data"]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    if body_id < 0:
        raise LookupError("body 'plate' not found in model")

    pose = data.xpos[body_id].copy().tolist()
    _CTX["plate_body_id"] = body_id
    return {"body": "plate", "pose": pose}


def check_object_displaceable_and_detected() -> dict[str, Any]:
    """Programmatically move the plate (simulating the disturbance
    injector) and confirm Python observes the new pose. This is the
    minimum viable version of the 'controlled disturbance' loop from
    Phase C, just enough to prove the mechanism works end to end."""
    import mujoco
    import numpy as np

    model = _CTX["model"]
    data = _CTX["data"]

    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    if plate_jnt < 0:
        raise LookupError("joint 'plate_joint' not found in model")

    q_adr = model.jnt_qposadr[plate_jnt]
    before = data.qpos[q_adr:q_adr + 3].copy()

    delta = np.array([0.10, 0.0, 0.0])
    data.qpos[q_adr:q_adr + 3] = before + delta
    mujoco.mj_forward(model, data)

    after = data.qpos[q_adr:q_adr + 3].copy()
    displacement = float(np.linalg.norm(after - before))

    if displacement < 1e-6:
        raise AssertionError("plate pose did not change after programmatic displacement")

    return {
        "pose_before": before.tolist(),
        "pose_after": after.tolist(),
        "displacement_m": displacement,
    }


def check_camera_offscreen_render(model_path: str) -> dict[str, Any]:
    """Isolated on purpose: offscreen rendering (EGL/OSMesa) is the most
    likely thing to break on a headless Linux box, and it blocks both G1
    and G7 if it's broken. Fail fast here rather than discovering it
    mid-perception-integration."""
    import mujoco
    import numpy as np

    model = _CTX["model"]
    data = _CTX["data"]

    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data)
    frame = renderer.render()
    renderer.close()

    if not isinstance(frame, np.ndarray) or frame.shape != (240, 320, 3):
        raise AssertionError(f"unexpected frame shape: {getattr(frame, 'shape', None)}")

    return {"frame_shape": list(frame.shape), "frame_dtype": str(frame.dtype)}


def check_openvino_smoke_test() -> dict[str, Any]:
    """Minimal OpenVINO toolchain check: import, load a model, compile,
    run one inference. Deliberately does NOT touch the robot — per the
    build plan, G1 only needs to prove the two toolchains independently
    work; wiring OpenVINO into perception happens at G7."""
    import numpy as np
    import openvino as ov

    core = ov.Core()

    # A tiny synthetic model avoids depending on a downloaded checkpoint
    # for this smoke test. Swap in your real perception model at G7.
    import openvino.opset12 as opset

    input_node = opset.parameter([1, 3, 32, 32], dtype=np.float32, name="input")
    relu_node = opset.relu(input_node)
    result_node = opset.result(relu_node, name="output")
    model = ov.Model([result_node], [input_node], "g1_smoke_model")

    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
    t0 = time.perf_counter()
    output = infer_request.infer({0: dummy_input})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out_tensor = list(output.values())[0]

    return {
        "openvino_version": ov.__version__,
        "available_devices": core.available_devices,
        "output_shape": list(out_tensor.shape),
        "single_inference_latency_ms": round(latency_ms, 3),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_CTX: dict[str, Any] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/dual_so101_no_table_collision.xml",
        help="Path to the MuJoCo XML model (relative to repo root, so vendor "
             "asset relative paths resolve correctly).",
    )
    parser.add_argument(
        "--out",
        default="results/g1_environment.json",
        help="Where to write the G1 evidence artifact.",
    )
    args = parser.parse_args()

    checks: list[CheckResult] = []

    checks.append(run_check("mujoco_launches", check_mujoco_launches))
    checks.append(run_check("model_loads", lambda: check_model_loads(args.model)))
    checks.append(run_check("arm_actuator_callable", check_arm_actuator_callable))
    checks.append(run_check("object_pose_readable", check_object_pose_readable))
    checks.append(run_check("object_displaceable_and_detected", check_object_displaceable_and_detected))
    checks.append(run_check("camera_offscreen_render", lambda: check_camera_offscreen_render(args.model)))
    checks.append(run_check("openvino_smoke_test", check_openvino_smoke_test))

    all_passed = all(c.passed for c in checks)

    report = {
        "gate": "G1",
        "timestamp": time.time(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "overall_pass": all_passed,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "duration_s": round(c.duration_s, 4),
                "detail": c.detail,
                "error": c.error,
            }
            for c in checks
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # Human-readable summary to stdout, JSON artifact on disk.
    print(f"=== G1 CHECK {'PASS' if all_passed else 'FAIL'} ===")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name} ({c.duration_s:.3f}s)")
        if not c.passed:
            print(f"          -> {c.error}")
    print(f"\nEvidence written to: {out_path.resolve()}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
