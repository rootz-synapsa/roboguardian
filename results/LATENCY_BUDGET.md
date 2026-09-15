# RoboGuardian — G8/H2 Latency Budget

Model: `models/dual_so101_g3.xml`  |  Seed: `42`  |  Trials: `20`  (0 required no recovery)

| Component | p50 (ms) | p95 (ms) | mean (ms) | max (ms) | n |
|---|---|---|---|---|---|
| Perception (initial observe) | 0.059 | 0.079 | 0.062 | 0.091 | 20 |
| Evaluation (initial classify) | 0.004 | 0.006 | 0.005 | 0.009 | 20 |
| Observe -> Decision | 0.063 | 0.085 | 0.066 | 0.100 | 20 |
| IK / Replan | 9.088 | 9.857 | 9.264 | 10.370 | 20 |
| Recovery execution (regrasp) | 10.205 | 10.585 | 10.267 | 10.936 | 20 |
| Recovery total | 19.649 | 20.299 | 19.763 | 20.975 | 20 |

**What this measures**: wall-clock time inside this Python/MuJoCo process on this machine -- ground-truth `SimulationStateProvider`, not the OpenVINO vision path (see `results/g7_openvino.json` for perception-model inference latency separately). No speed claim is made here; the goal is to show the control loop's own overhead is small relative to the disturbance-recovery task, not to claim it is fast in an absolute sense.
