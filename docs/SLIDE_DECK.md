# RoboGuardian — Slide Deck Structure (G8)

10 slides. Every quantitative bullet cites its artifact in speaker notes.

1. **Title** — RoboGuardian; thesis line (resilience wording); repo URL; team.
2. **Problem** — open-loop blindness: plan once, execute blind; image = Arm B end state.
   Notes: g3_failure.json (plate 0.30 m off target, 10/10 reproducible).
3. **Approach** — Observe → Evaluate → Re-plan → Execute loop diagram.
4. **Section 18 guarantee** — code snippet: stale target logged, never issued.
   Notes: structural, not runtime-checked; traces show stale_action_executed=false.
5. **Experiment design** — arms A/B/C; disturbance spec (+0.10 m X @ before_grasp);
   thresholds (normal 0.01/0.03 m, recover 0.15 m); deterministic seeds.
6. **Results** — table: A 10/10, B 0/10, C 10/10; G4 100%/0%; G5 100%.
   Notes: g6_reliability.json, g4_detection*.json, g5_recovery*.json.
7. **Vision (G7)** — accuracy ~0.7 cm (worst 2.4 cm), latency <2 ms;
   OpenVINO 0.44x vs ONNX Runtime — stated plainly.
8. **Evidence & reproducibility** — artifact tree; seeds; one-command reruns.
9. **Limitations** — kinematic carry; deterministic disturbance; single
   disturbance type/target; CPU-only; no statistical test.
10. **Future work + links** — stochastic disturbance; end-to-end vision recovery;
    real hardware; repo + video links.

Speaker-note rule: never say "faster", "proof", or "dominates".

## Final Assets (2026-09-15)

- Deck PDF: `presentation/RoboGuardian_Governed_Execution_Deck_v2.pdf`
- Editable source: `presentation/RoboGuardian_Governed_Execution_Deck_v2.pptx`
- Frames: `presentation/slides/`
