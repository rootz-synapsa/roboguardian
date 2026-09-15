# Reproducibility

## Environment

This project pins exact dependency versions in `requirements-lock.txt`,
generated from the working development environment via:

```bash
pip freeze > requirements-lock.txt
```

`requirements.txt` lists unpinned package names for readability;
`requirements-lock.txt` is the one to use for exact reproduction:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
```

Offscreen rendering (MuJoCo `Renderer`) requires OSMesa or EGL on
headless machines:

```bash
export MUJOCO_GL=osmesa   # or: egl
```

## Regenerating evidence, gate by gate

All commands assume the repo root as the working directory.

| Gate | Command | Artifact |
|---|---|---|
| G1 | `python experiments/g1_check.py --model models/dual_so101_g3.xml` | `results/g1_environment.json` |
| G3 | `python experiments/run_baseline_failure.py --model models/dual_so101_g3.xml --seed 42 --trials 10` | `results/g3_failure.json` |
| G4 | `python experiments/run_detection_check.py --model models/dual_so101_g3.xml --seed 42 --trials 10` | `results/g4_detection.json` |
| G5 | `python experiments/run_recovery.py --model models/dual_so101_g3.xml --seed 42 --trials 10 --provider simulation` | `results/g5_recovery.json` |
| G6 | `python experiments/run_g6_reliability.py --trigger before_grasp --delta 0.10 0.0 0.0 --seed 42 --trials 10` | `results/g6_reliability.json`, `results/RESULTS.md` |
| G7 | `python perception/dataset_gen.py && python perception/train_detector.py && python experiments/benchmark_openvino.py` | `results/g7_openvino.json` |
| G8/H1 | `python experiments/run_robustness_matrix.py --seed 42 --trials 10` | `results/g8_robustness_matrix.json`, `results/ROBUSTNESS_MATRIX.md` |
| G8/H2 | `python experiments/benchmark_control_loop.py --model models/dual_so101_g3.xml --seed 42 --trials 20` | `results/g8_latency_budget.json`, `results/LATENCY_BUDGET.md` |

All disturbance-based runs use `seed=42` and `noise_std=0`: results are
deterministic and should reproduce bit-for-bit on the same MuJoCo
version and platform. Small floating-point differences across
OS/CPU/MuJoCo-version combinations are possible but should not change
any PASS/FAIL classification, since thresholds carry meaningful margin
(see each gate's `config` block in its JSON artifact).

## What CI does and does not cover

`.github/workflows/ci.yml` runs on every push/PR:
1. Syntax check (`python -m compileall .`)
2. Unit + recovery invariant tests (`pytest -q`, see `tests/test_recovery_invariants.py`)
3. G1 environment/toolchain smoke test

CI deliberately does **not** run G6 (10+ trials), G8's robustness
matrix (80-90 runs), or OpenVINO benchmarking — those need stable
timing measurement that a shared CI runner cannot guarantee, and their
results are committed as evidence artifacts under `results/` instead
of re-derived on every push. Run them locally when the recovery logic,
model, or thresholds change.

## Recovery invariants under test

`tests/test_recovery_invariants.py` checks the guarantees the project's
core claim depends on, independent of any specific trial's outcome:

- `test_stale_target_never_written_to_ctrl` — the pre-disturbance joint
  target is never sent to an actuator, checked by comparing final
  `ctrl` state directly against the stale target's values.
- `test_recovery_budget_is_bounded` — `recovery_attempts_used` and
  `reobservations_used` never exceed their configured maximums.
- `test_failed_ik_never_executes` — a non-converged IK solution is
  never executed, verified via a monkeypatched `solve_arm_ik`.
- `test_fresh_observation_required_before_recovery` — `observe()` is
  called at least once per recovery attempt before any action.
- `test_uncertain_state_safe_stops` — exhausting the reobservation
  budget while stuck at `UNCERTAIN` always resolves to `SAFE_STOP`,
  never a guess.

These tests use the real project model (`models/dual_so101_g3.xml`)
for realistic integration coverage, not a synthetic mock.
