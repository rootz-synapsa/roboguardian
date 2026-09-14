# RoboGuardian — 5-Minute Video Outline (G8)

| Time | Section | Screen | Caption / voice |
|---|---|---|---|
| 0:00–0:30 | Hook | Arm B end-state render (plate 0.30 m off target) | "One nudge. The plan becomes fiction." |
| 0:30–1:00 | Thesis | README header | "Governed execution is more resilient than open-loop execution when the world changes." |
| 1:00–1:40 | Loop | Architecture diagram | Observe → Evaluate → Re-plan → Execute; Section 18: stale plan never executed |
| 1:40–2:40 | Baseline fails | Terminal: `run_baseline_failure.py --trials 1` | detection-free baseline grasps thin air; plate left behind |
| 2:40–3:40 | Guardian recovers | Terminal: `run_recovery.py --trials 1` + event trace | pause → fresh IK → regrasp → verify → resume |
| 3:40–4:20 | Head-to-head | `RESULTS.md` table | A 10/10 · B 0/10 · C 10/10 (deterministic, 10 trials/arm) |
| 4:20–4:45 | Vision | `run_detection_check.py --provider vision` | camera-only detection 100%/0% FP; error ~0.7 cm (worst 2.4 cm) |
| 4:45–5:00 | Honesty + repo | Limitations list + repo URL | kinematic carry, deterministic disturbance, OpenVINO 0.44x — reported as measured |

Recording checklist:
- [ ] Terminal zoomed (font ≥ 18 pt), dark theme, no secrets/paths with usernames
- [ ] Screen region: terminal 70% + results pane 30%
- [ ] Captions burned in for the killer-demo line
- [ ] Voice-over recorded after screen capture (easier retakes)
- [ ] Export 1080p30, ≤ 5:00 hard cut
