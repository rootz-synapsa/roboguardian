# RoboGuardian — Project Description (Submission)

**One-liner:** Governed execution is more resilient than open-loop
execution when the world changes — demonstrated in a bimanual dual
SO-101 MuJoCo simulation with an OpenVINO vision path.

**Problem.** Open-loop pick-and-place plans assume a static world. When
the object moves mid-execution, the plan becomes invalid but execution
continues: in our tests the baseline failed 10/10 deterministic trials
(plate left 0.30 m from target).

**Method.** RoboGuardian wraps execution in a governance loop:
Observe (state provider: ground truth or OpenVINO vision) → Evaluate
(3-state classifier with configurable thresholds) → Re-plan (fresh IK
from fresh observation) → Execute. Section 18 guarantee: the stale
target is accepted only as evidence to log and is structurally never
sent to actuators.

**Evaluation.** Three arms, 10 deterministic trials each, identical
seeds/disturbance: A (no disturbance) 10/10; B (disturbance, open-loop)
0/10; C (disturbance + governance) 10/10. Detection: 100% with 0% false
recovery (ground truth and vision modes). Recovery success 100% with
`stale_action_executed=false` in every trace. Vision: ~0.7 cm mean error
(worst 2.4 cm), <2 ms inference on CPU.

**Honesty.** This is controlled simulation evidence, not a formal proof:
deterministic trials, single disturbance type and target, kinematic carry
grasping, CPU-only. OpenVINO measured 0.44x of ONNX Runtime on this tiny
model — reported as measured, no speedup claimed.

**Reproducibility.** All gates rerun via single commands (README Quick
Start); artifacts in `results/` (JSON + JSONL traces); full analysis in
`results/FINAL_REPORT.md`; demo assets in `docs/`.

**Repo:** https://github.com/rootz-synapsa/roboguardian
