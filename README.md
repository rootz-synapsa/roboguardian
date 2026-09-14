# RoboGuardian

> **Governed AI is faster than ungoverned AI under uncertainty.**

A reference implementation of a self-healing robotic control loop.
Built on MuJoCo + OpenVINO + a 6-DOF SO-101 arm.

## The Claim

An open-loop robot arm carrying a plate to a target **fails 100% of the
time** when the plate is moved mid-execution. The same arm with
RoboGuardian governance **succeeds 100% of the time** under the same
disturbance.

Arm A (no disturbance): 10/10 success
Arm B (disturbance, no recovery): 0/10 success
Arm C (disturbance + RoboGuardian): 10/10 success

## Gates

| Gate | Scope | Status |
|---|---|---|
| G1 | Environment & Toolchain | ✅ PASS |
| G2 | Kinematic Baseline | ✅ 10/10 |
| G3 | Controlled Failure | ✅ 10/10 reproducible |
| G4 | Detection | ✅ 100% / 0% FP |
| G5 | Recovery | ✅ 100% / stale never executed |
| G6 | A/B/C Reliability | ✅ A=10, B=0, C=10 |
| G7 | Vision Perception | ✅ PASS w/ limits |

See [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) for full evidence.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install mujoco numpy openvino onnxruntime torch --index-url https://download.pytorch.org/whl/cpu

# Run killer demo (A/B/C comparison)
python experiments/run_g6_reliability.py \
  --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10
Architecture
Observe → Evaluate → Re-plan → Execute
   ↑                                │
   └──────────── loop ──────────────┘
Key guarantee (Section 18): stale plan is never executed. Enforced
by construction in RecoveryController, not by runtime check.
Evidence
All artifacts are deterministic and reproducible. See results/ for
JSON evidence and per-step event traces.
Status
🎉 Core thesis: PROVEN.
