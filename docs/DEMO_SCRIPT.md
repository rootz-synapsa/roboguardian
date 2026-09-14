# RoboGuardian — Live Demo Script (G8)

Target: 8–10 min live demo (also the shooting script for the 5-min video).
Rule: speak only claims listed in the Claim↔Evidence map at the end.

## 0. Pre-flight (before audience)
- [ ] `source .venv/bin/activate`
- [ ] `python -c "import mujoco, openvino, onnxruntime, torch; print('deps ok')"`
- [ ] Files exist: `models/dual_so101_g3.xml`, `results/plate_detector.onnx`,
      `results/plate_detector_norm.json`
- [ ] Warm caches: run each demo command once privately
- [ ] Two terminal panes: left = commands/output, right = `results/RESULTS.md`
- [ ] Fallback: pre-recorded capture + pre-generated JSONs in `results/`

## 1. Hook (60 s) — "The world changes; plans don't."
Say: an open-loop robot plans once and executes blind; one nudge to the
object turns the plan into fiction.
Run: `python experiments/run_baseline_failure.py --trials 1 --out /tmp/demo_g3.json`
Show: plate ends 0.30 m from target; `task_failed=true`.
Point at: `results/g3_failure.json`.

## 2. Thesis + loop (60 s)
Say (exact wording): "Governed execution is more resilient than open-loop
execution when the world changes." Do NOT say "faster".
Show: architecture diagram from README (Observe → Evaluate → Re-plan → Execute).

## 3. Governed run (2–3 min) — Arm C
Run: `python experiments/run_recovery.py --provider simulation --trials 1 --out /tmp/demo_c.json`
Narrate: detection at `before_grasp` (RECOVERABLE) → pause → fresh IK from
fresh observation → regrasp → verify → resume.
Show trace: `tail -n 5 results/events/G5-RECOVERY-simulation-seed42.jsonl`
Point at: `"stale_action_executed": false` (Section 18, by construction).

## 4. Head-to-head (2 min) — A/B/C
Run: `python experiments/run_g6_reliability.py --trials 10`
Show table + killer line: A 10/10, B 0/10, C 10/10 (`results/RESULTS.md`).

## 5. Perception without ground truth (1–2 min, optional)
Run: `python experiments/run_detection_check.py --provider vision --normal-threshold 0.03 --trials 10 --out /tmp/demo_vision.json`
Say: camera-only detection 100% / 0% FP at 0.03 m threshold; vision error
~0.7 cm mean, worst 2.4 cm at workspace edge (state the limit out loud).

## 6. Close (30 s)
Limitations out loud: kinematic carry, deterministic disturbance, single
disturbance type/target, CPU-only, OpenVINO 0.44x (no speedup claimed).
Repo: https://github.com/rootz-synapsa/roboguardian

## Claim ↔ Evidence map
| Say | Backed by |
|---|---|
| baseline fails 10/10 under disturbance | `results/g3_failure.json`, G6 Arm B |
| governed recovers 10/10 | `results/g5_recovery*.json`, G6 Arm C |
| detection 100% / FP 0% | `results/g4_detection*.json` |
| stale plan never executed | recovery traces `stale_action_executed=false` |
| vision works without ground truth | `results/g4_detection_vision_t03.json` |
| decision latency in ms | `results/g6_reliability.json` (`mean_decision_latency_ms`) |
| NEVER say "faster" | G7: OpenVINO 0.44x vs ONNX Runtime |
