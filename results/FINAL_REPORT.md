# RoboGuardian — Final Report

> **"Governed execution is more resilient than open-loop execution when the world changes."**
>
> Controlled simulation evidence that a governed control loop with
> observe → evaluate → re-plan semantics outperformed an open-loop
> baseline when the world changes mid-execution.

---

## Executive Summary

RoboGuardian is a reference implementation of a **self-healing robotic
control loop** built around three primitives:

1. **Observe** — detect that the world has drifted from the plan
2. **Evaluate** — classify the drift (normal / recoverable / unrecoverable)
3. **Re-plan** — solve a fresh action from fresh observation, never from
   the stale plan

Built on MuJoCo (simulation), OpenVINO (perception inference), and a
bimanual dual SO-101 simulation environment (experiments exercise the left arm manipulating a plate on a table).

**Result:** across 10 trials per arm,

| Arm | Description | Success |
|---|---|---|
| **A** | No disturbance (happy path) | **10 / 10** |
| **B** | Disturbance + open-loop baseline | **0 / 10** |
| **C** | Disturbance + RoboGuardian | **10 / 10** |

The governed arm (C) recovers 100% of the time; the ungoverned arm (B)
fails 100% of the time (10 deterministic trials per arm).

---

## Core Thesis

H0 (null): Governed control offers no advantage over open-loop.
H1 (thesis): Governed control is more resilient than open-loop control under uncertainty.

**Evidence chain:**

| Gate | What we proved | Result |
|---|---|---|
| G1 | Toolchain (MuJoCo + OpenVINO) works | PASS |
| G2 | Baseline succeeds on a static world | 10/10 ✅ |
| G3 | Baseline fails reproducibly under disturbance | 10/10 FAIL ✅ |
| G4 | Guardian detects disturbance, low false positive | 100% / 0% ✅ |
| G5 | Guardian recovers; stale plan never executed | 100% / structural ✅ |
| G6 | A/B/C apples-to-apples comparison | A=10, B=0, C=10 ✅ |
| G7 | Vision-based perception replaces ground truth | PASS (w/ limits) ✅ |

**Result: the evidence supports H1 under the tested scenario. This is controlled simulation evidence, not a formal proof; no statistical test was performed (trials are deterministic re-runs).**

---

## Architecture


**Evidence chain:**

| Gate | What we proved | Result |
|---|---|---|
| G1 | Toolchain (MuJoCo + OpenVINO) works | PASS |
| G2 | Baseline succeeds on a static world | 10/10 ✅ |
| G3 | Baseline fails reproducibly under disturbance | 10/10 FAIL ✅ |
| G4 | Guardian detects disturbance, low false positive | 100% / 0% ✅ |
| G5 | Guardian recovers; stale plan never executed | 100% / structural ✅ |
| G6 | A/B/C apples-to-apples comparison | A=10, B=0, C=10 ✅ |
| G7 | Vision-based perception replaces ground truth | PASS (w/ limits) ✅ |

**Result: the evidence supports H1 under the tested scenario. This is controlled simulation evidence, not a formal proof; no statistical test was performed (trials are deterministic re-runs).**

---

## Architecture


### Components

| File | Role |
|---|---|
| `sim/disturbance.py` | Deterministic disturbance injector |
| `perception/state_provider.py` | Ground-truth observation |
| `perception/openvino_provider.py` | Vision observation (OpenVINO) |
| `perception/dataset_gen.py` | Synthetic dataset generator |
| `perception/train_detector.py` | Tiny CNN training (PyTorch) |
| `control/state_machine.py` | 5-state evaluator |
| `control/ik_solver.py` | Damped least-squares IK |
| `control/recovery.py` | Section-18 compliant recovery |
| `evidence/event_logger.py` | Append-only JSONL trace |

---

## Evidence Summary

### G1 — Environment & Toolchain

- MuJoCo 3.x offscreen render: **OK**
- OpenVINO 2026.3.1 CPU compile: **OK**
- PyTorch CPU wheel: **OK**
- Artifact: `results/g1_environment.json`

### G2 — Kinematic Baseline

- 10/10 trials PASS
- Final XY error: 0.0451 m (tolerance: 0.05 m)
- Artifact: `results/g2_baseline.json`

### G3 — Controlled Failure

- 10/10 trials FAIL reproducibly
- Plate final XY error: 0.30 m (target missed by 30 cm)
- Artifact: `results/g3_failure.json`

### G4 — Detection

| Provider | Detection Rate | False Positive |
|---|---|---|
| Simulation (ground truth) | 100% | 0% |
| Vision (OpenVINO, threshold 0.03m) | 100% | 0% |

- Artifacts:
  - `results/g4_detection.json`
  - `results/g4_detection_vision_t03.json`

### G5 — Recovery

| Provider | Recovery Rate | Stale Executed |
|---|---|---|
| Simulation | 100% | **False** (always) |
| Vision | 100% | **False** (always) |

- Final XY error after recovery: 0.0451 m (same as G2)
- Artifacts:
  - `results/g5_recovery.json`
  - `results/g5_recovery_vision.json`

### G6 — A/B/C Reliability

| Arm | Success | Stale Executed |
|---|---|---|
| A — No disturbance | 10 / 10 (100%) | N/A |
| B — Disturbance, no recovery | 0 / 10 (0%) | 10 / 10 (100%) |
| C — Disturbance + RoboGuardian | 10 / 10 (100%) | 0 / 10 (0%) |

- Artifact: `results/g6_reliability.json`

### G7 — Perception Integration

| Metric | Value |
|---|---|
| Vision accuracy (mean error) | ~0.7 cm |
| Vision latency (p50) | 0.5 – 1.2 ms |
| OpenVINO vs ONNXRuntime speedup | 0.44x (no speedup) |
| G4 w/ vision | 100% / 0% ✅ |
| G5 w/ vision | 100% / structural ✅ |

- Artifacts:
  - `results/g7_openvino.json`
  - `results/plate_detector.onnx`
  - `results/plate_detector_norm.json`

---

## Killer-Demo Summary Line

Arm A (no disturbance): 10/10 success
Arm B (disturbance, no recovery): 0/10 success
Arm C (disturbance + RoboGuardian): 10/10 success

Use this verbatim in README, slides, and video captions.

---

## Section 18 — Structural Guarantee

The most important invariant in the system:

> **"Stale target = never issued."**

This is enforced **by construction**, not by runtime check:

```python
def recover(self, stale_target, expected_pose):
    trace = RecoveryTrace(stale_target=dict(stale_target))  # log only
    ...
    # stale_target is NEVER written to data.ctrl
    # Only fresh IK solution is executed
    self._execute_arm(trace.fresh_target, GRIPPER_CLOSE_VALUE, steps=60)
The stale_target parameter exists only to be logged in the
evidence trace. No code path in RecoveryController ever sends it to
an actuator. Verified empirically: stale_target_never_executed = True
across all trials in G5 and G6.
Limitations (Honest Accounting)
Per section 3 (no overclaim), the following are known limits:
Simulation fidelity
Kinematic carry model: plate is teleported to follow the gripper
every step; no physical grasping force or contact dynamics.
Deterministic disturbance: noise_std=0 in all trials; stochastic
robustness not yet measured.
Self-contact ~1.3 mm: present in G2 baseline but not blocking.
Vision system
Position-dependent error: best 0.4 cm (center), worst 2.4 cm (edge).
Normal threshold adjustment: raised from 0.01 → 0.03 m to
accommodate vision noise.
No OpenVINO speedup: 0.44x on CPU. Model is too small (~146K params)
for OpenVINO's optimization overhead to pay off. Absolute latency
(<2 ms) is still well within control loop budget, so this is not a
functional problem.
Recovery architecture
Hybrid perception: initial detection uses the selected provider
(sim or vision), but RecoveryController.recover() internally uses
SimulationStateProvider (ground truth) for its re-observation.
A full end-to-end vision recovery would require passing the provider
into the controller.
Single disturbance type: only object_move tested. Occlusion,
lighting change, object swap not yet exercised.
Single target position: only [-0.2, 0.0, 0.43] validated.
Physical deployment
No real hardware tested.
No safety-certified runtime (this is a research prototype).
Reproducibility
All evidence is deterministic (same seed → same result). To reproduce:
source .venv/bin/activate

# G2 baseline
python experiments/run_g2_baseline.py

# G3 controlled failure
python experiments/run_baseline_failure.py \
  --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10

# G4 detection (ground truth)
python experiments/run_detection_check.py \
  --provider simulation --trials 10

# G4 detection (vision)
python experiments/run_detection_check.py \
  --provider vision --normal-threshold 0.03 \
  --onnx-model results/plate_detector.onnx \
  --norm-stats results/plate_detector_norm.json \
  --out results/g4_detection_vision_t03.json

# G5 recovery (vision)
python experiments/run_recovery.py \
  --provider vision --normal-threshold 0.03 \
  --onnx-model results/plate_detector.onnx \
  --norm-stats results/plate_detector_norm.json \
  --out results/g5_recovery_vision.json

# G6 A/B/C reliability
python experiments/run_g6_reliability.py \
  --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10

# G7 vision pipeline
python perception/dataset_gen.py --n 800
python perception/train_detector.py --data results/g7_dataset
python experiments/benchmark_openvino.py --onnx results/plate_detector.onnx
Artifact Index
results/
├── g1_environment.json
├── g2_baseline.json
├── g3_failure.json
├── g4_detection.json
├── g4_detection_vision_t03.json
├── g5_recovery.json
├── g5_recovery_vision.json
├── g6_reliability.json
├── g7_openvino.json
├── g7_dataset/
│   ├── labels.json
│   └── images/00000.npy ... 00799.npy
├── plate_detector.onnx
├── plate_detector_norm.json
├── RESULTS.md
├── FINAL_REPORT.md
└── events/
    ├── G3-BASELINE-seed42.jsonl
    ├── G4-DETECTION-simulation-seed42.jsonl
    ├── G4-DETECTION-vision-seed42.jsonl
    ├── G5-RECOVERY-simulation-seed42.jsonl
    ├── G5-RECOVERY-vision-seed42.jsonl
    └── G6-RELIABILITY-seed42.jsonl
Future Work
Stochastic disturbance — noise_std > 0 trials to measure
robustness under uncertainty.
End-to-end vision recovery — pass provider into
RecoveryController so re-observation also uses vision.
Multi-object scenes — test detection/recovery with multiple
distractors.
Real hardware — port to physical SO-101 + RealSense camera.
Larger vision model — MobileNetV2-class to see if OpenVINO
speedup materializes.
Conclusion
RoboGuardian demonstrates, with reproducible evidence, that a governed
control loop with observe/evaluate/re-plan semantics outperformed
an open-loop baseline when the world changes mid-execution.
The system is small (~146K params for vision, ~1K lines of Python),
runs on CPU, and every claim is backed by a JSON artifact.
Core thesis: SUPPORTED by controlled simulation evidence. ✅
Report generated: 2026-09-15
Repository: ~/Projects/roboguardian


## Scope of Validity

All quantitative claims hold under the tested configuration only:
single disturbance type (`object_move`, +0.10 m in X, fired at
`before_grasp`), single place target, deterministic seeds
(`noise_std=0`), kinematic carry grasping model, CPU execution, MuJoCo
simulation. "10/10" expresses reproducibility of deterministic re-runs,
not a statistical confidence interval.

## Claim Revision Log

| Date | Change | Reason |
|---|---|---|
| 2026-09-15 | "formal proof" -> "controlled simulation evidence" | Founder review: 10 deterministic trials cannot constitute a proof |
| 2026-09-15 | "H0 rejected / H1 accepted" -> "supports H1 under tested scenario" | same as above |
| 2026-09-15 | "strictly dominates" -> "outperformed" | same as above |
| 2026-09-15 | thesis "faster" -> "more resilient" (README + FINAL_REPORT) | evidence measures success/recovery, not speed; G7 shows OpenVINO 0.44x |
| 2026-09-15 | "6-DOF SO-101 arm" -> "bimanual dual SO-101 simulation" | match actual repo infrastructure and challenge framing |
