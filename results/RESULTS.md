# RoboGuardian — G6 Reliability Results

Model: `models/dual_so101_g3.xml` | Trigger: `before_grasp` | Delta: `[0.1, 0.0, 0.0]` | Seed: `42` | Trials/arm: `10`

| Arm | Success | Detection | Recovery | Stale-Exec | Safe-Stop | Latency (ms) | Time (s) |
|---|---|---|---|---|---|---|---|
| A — No disturbance | 10/10 (100%) | n/a | n/a | 0% | 0% | n/a | 0.75 |
| B — Disturbance, no recovery | 0/10 (0%) | 0% | n/a | 100% | 0% | n/a | 0.33 |
| C — Disturbance + recovery | 10/10 (100%) | 100% | 100% | 0% | 0% | 0.07 | 0.68 |

## Killer-demo summary line

- Arm A (no disturbance): 10/10 success
- Arm B (disturbance, no recovery): 0/10 success
- Arm C (disturbance + RoboGuardian): 10/10 success
