"""
G7 — OpenVINO benchmark.

Implements the benchmark protocol from build-plan section 21: warm-up
before measuring, initialization/compile time excluded from inference
latency, p50/p95/mean/max reported, environment metadata recorded.

Baseline = ONNXRuntime CPU execution of the same ONNX graph. This is a
fair baseline (identical model, identical hardware, identical input) --
not a strawman.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov

SPEEDUP_CLAIM_MARGIN = 1.05


def bench_onnxruntime(
    onnx_path: str,
    n_warmup: int,
    n_trials: int,
    img_size: int,
) -> list[float]:
    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(
        1, 3, img_size, img_size
    ).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def bench_openvino(
    onnx_path: str,
    n_warmup: int,
    n_trials: int,
    img_size: int,
    device: str,
) -> list[float]:
    core = ov.Core()
    compiled = core.compile_model(onnx_path, device)
    infer_request = compiled.create_infer_request()
    dummy = np.random.rand(
        1, 3, img_size, img_size
    ).astype(np.float32)

    for _ in range(n_warmup):
        infer_request.infer({0: dummy})

    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        infer_request.infer({0: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def summarize(latencies: list[float]) -> dict:
    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx", default="results/plate_detector.onnx"
    )
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--out", default="results/g7_openvino.json"
    )
    args = parser.parse_args()

    baseline_latencies = bench_onnxruntime(
        args.onnx, args.warmup, args.trials, args.img_size
    )
    ov_latencies = bench_openvino(
        args.onnx, args.warmup, args.trials, args.img_size,
        args.device,
    )

    baseline_summary = summarize(baseline_latencies)
    ov_summary = summarize(ov_latencies)

    speedup_p50 = (
        baseline_summary["p50_ms"] / ov_summary["p50_ms"]
        if ov_summary["p50_ms"] > 0
        else None
    )

    claim_supported = (
        speedup_p50 is not None
        and speedup_p50 >= SPEEDUP_CLAIM_MARGIN
    )

    claim = (
        "OpenVINO enables a faster closed-loop response."
        if claim_supported
        else "OpenVINO is used to optimize perception inference."
    )

    report = {
        "gate": "G7",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "openvino_version": ov.__version__,
            "onnxruntime_version": ort.__version__,
            "device": args.device,
        },
        "config": {
            "onnx_model": args.onnx,
            "img_size": args.img_size,
            "warmup": args.warmup,
            "trials": args.trials,
        },
        "baseline_onnxruntime_cpu": baseline_summary,
        "openvino": ov_summary,
        "speedup_p50_x": speedup_p50,
        "claim_supported": claim_supported,
        "claim": claim,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(
        f"Baseline (ONNXRuntime CPU) p50: "
        f"{baseline_summary['p50_ms']:.3f} ms  "
        f"(p95={baseline_summary['p95_ms']:.3f})"
    )
    print(
        f"OpenVINO p50:                  "
        f"{ov_summary['p50_ms']:.3f} ms  "
        f"(p95={ov_summary['p95_ms']:.3f})"
    )
    if speedup_p50:
        print(f"Speedup (p50): {speedup_p50:.2f}x")
    else:
        print("Speedup: n/a")
    print(f"Claim: {claim}")
    print(f"Evidence written to: {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
