# RoboGuardian

> **Governed execution is more resilient than open-loop execution when the world changes.**

A reference implementation of a self-healing robotic control loop.
Built on MuJoCo + OpenVINO in a **bimanual dual SO-101 simulation
environment**; the current evidence chain exercises the left-arm
pick-and-place task.

## The Claim — and what we do NOT claim

When the target object is moved mid-execution, an open-loop arm
**fails 10/10** deterministic trials. The same arm with RoboGuardian
governance **succeeds 10/10** under the identical disturbance.

We do **not** claim governed execution is *faster*. Governance adds an
observe/evaluate step (milliseconds), and our OpenVINO CPU inference
measured **0.44x** of ONNX Runtime on this tiny model (no speedup
claimed). The claim is **resilience under world change**, backed by
controlled simulation evidence — not a formal proof.

## Killer demo (10 deterministic trials per arm)

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

See [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) for full evidence
and [`docs/`](docs/) for the submission pack.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install mujoco numpy openvino onnxruntime torch --index-url https://download.pytorch.org/whl/cpu

# Killer demo (A/B/C comparison)
python experiments/run_g6_reliability.py \
  --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10
Architecture

Observe -> Evaluate -> Re-plan -> Execute
   ^                                |
   +------------ loop --------------+

Key guarantee (Section 18): the stale plan is never executed —
enforced by construction in RecoveryController (the stale target is
accepted only as a value to log), not by runtime check.
Evidence & honesty notes
All trials are deterministic (fixed seeds, noise_std=0): "10/10"
expresses reproducibility, not a statistical confidence interval.
Tested scope: single disturbance type (object_move, +0.10 m in X at
before_grasp), single place target, kinematic carry grasping model.
Vision (G7): ~0.7 cm mean position error, worst 2.4 cm at workspace
edge; detection threshold raised 0.01 -> 0.03 m in vision mode.
OpenVINO vs ONNX Runtime: 0.44x (slower) on this ~146K-param model.
Reported as measured; absolute latency (<2 ms) is still within budget.
Status
✅ G1–G7 closed. Core thesis supported by controlled simulation
evidence under the tested scenario. Submission pack in docs/.

## Submission Assets (G8)

- Video (2:40 ≤ 5:00): [presentation/roboguardian_pitch.mp4](presentation/roboguardian_pitch.mp4)
- Slide deck (PDF): [presentation/RoboGuardian_Governed_Execution_Deck_v2.pdf](presentation/RoboGuardian_Governed_Execution_Deck_v2.pdf)
- Slide frames + ffmpeg manifest: [presentation/slides/](presentation/slides/) + [presentation/slides_config.txt](presentation/slides_config.txt)
