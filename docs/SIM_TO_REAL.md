# Sim-to-Real Contract (G8-H4)

What changes and what does NOT when moving from MuJoCo to a real SO-101.

## 1. State Provider Contract
observe(object_name, expected_pose) -> WorldState
- Simulation: SimulationStateProvider (MuJoCo xpos)
- Real: camera -> OpenVINO -> VisionStateProvider
- StateEvaluator is provider-agnostic (never knows the source).

## 2. Execution Contract
FreshAction -> RobotAdapter -> actuators
- Simulation: MuJoCo data.ctrl (position actuators)
- Real: SO-101 motor driver via RobotAdapter (same joint-order contract)

## 3. Safety Boundary (unchanged across sim/real)
- joint/velocity/force limits (frozen G2 values)
- workspace limits; MAX_RECOVERY_ATTEMPTS=2; MAX_REOBSERVATION_ATTEMPTS=2
- SAFE_STOP on budget exhaustion or UNRECOVERABLE
- Section 18: stale target never reaches actuators (structural)

## 4. Hardware Mapping
camera -> OpenVINO perception -> StateProvider -> StateEvaluator
-> RecoveryController -> RobotAdapter -> SO-101

## 5. Explicit Non-Claims
- Not tested on real hardware
- Not safety certified
- No sim-to-real validation yet
- confidence field is a placeholder (1.0); UNCERTAIN path not yet meaningful
