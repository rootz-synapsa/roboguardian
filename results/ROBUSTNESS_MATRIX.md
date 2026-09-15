# RoboGuardian — G8/H1 Robustness Matrix

Model: `models/dual_so101_g3.xml`  |  Seed: `42`  |  Trials/condition: `10`

| Condition | Success | Safe-Stop | Stale-Exec | Attempts (mean) | Pass |
|---|---|---|---|---|---|
| baseline | 100% | 0% | 0% | 2.00 | ✅ |
| displacement_5cm | 100% | 0% | 0% | 2.00 | ✅ |
| displacement_15cm | 100% | 0% | 0% | 2.00 | ✅ |
| noise_1cm | 100% | 0% | 0% | 2.00 | ✅ |
| noise_2cm | 60% | 40% | 0% | 2.00 | ✅ |
| delay_50ms_proxy | 100% | 0% | 0% | 2.00 | ✅ |
| delay_100ms_proxy | 100% | 0% | 0% | 2.00 | ✅ |
| dropped_frame | 100% | 0% | 0% | 2.00 | ✅ |

**Pass criterion**: `stale_action_execution_rate == 0%` AND every trial reached a valid terminal outcome (success or a correctly-reasoned SAFE_STOP). Raw task success rate is reported for context, not used as the pass bar -- a governed SAFE_STOP under a disturbance that exceeds the recovery budget is a correct outcome, not a failure.

Overall G8/H1: PASS
