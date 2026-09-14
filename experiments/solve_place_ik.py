"""
Re-solve PLACE joint targets via IK on the corrected scene
============================================================

Context: shoulder_lift / elbow_flex force saturation went away once their
force limits were raised (3.35 -> 7.5 / 10.0). wrist_flex is still
saturating. Before raising its limit too, this script tests the more
likely root cause: the PLACE joint targets were solved (or hand-tuned)
against the *dirty* scene — before plate<->gripper and target_object
legacy collisions were disabled. If the geometry changed, those old
joint angles may require excess torque to reach a pose that no longer
matches where place_target actually sits (or matches, but via an
unnatural configuration that overloads the wrist).

This script re-solves the arm's joint angles for the PLACE step using
damped least-squares IK on the corrected model, targeting the
`left_gripperframe` site to reach the `place_target` body's current
position (and, optionally, orientation). It does NOT touch force limits.

Workflow:
  1. Run this after all collision fixes (including the 3 remaining
     gripper<->table contacts) are applied to the model.
  2. Take the new joint solution it prints and swap it in for PLACE in
     your diagnostic/execution script.
  3. Re-run the diagnostic. If wrist_flex saturation disappears, the
     root cause was stale IK, not an undersized force limit.
  4. If it *still* saturates against a freshly-solved target, that's a
     real physical/limit question worth raising WRIST_FORCE_LIMIT for
     -- with the corrected target as justification, not a copy of the
     shoulder/elbow numbers.

Usage:
    python experiments/solve_place_ik.py --model models/dual_so101_no_table_collision.xml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    # gripper excluded from IK -- it's an end-effector open/close DOF,
    # not part of reaching the target position.
]

SITE_NAME = "left_gripperframe"
TARGET_BODY_NAME = "place_target"

MAX_ITERS = 200
DLS_DAMPING = 0.05
POS_TOL_M = 1e-4
STEP_CLAMP_RAD = 0.2  # avoid huge single-step jumps that overshoot into new collisions


def solve_ik(model_path: str, seed_qpos: dict[str, float] | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise LookupError(f"site not found: {SITE_NAME}")

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    if target_body_id < 0:
        raise LookupError(f"body not found: {TARGET_BODY_NAME}")

    joint_ids = []
    qpos_adrs = []
    dof_adrs = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{name}")
        if jid < 0:
            raise LookupError(f"joint not found: left_{name}")
        joint_ids.append(jid)
        qpos_adrs.append(model.jnt_qposadr[jid])
        dof_adrs.append(model.jnt_dofadr[jid])

    # Optional seed (e.g. the current LIFT pose) so IK converges to a
    # solution *near* the existing trajectory instead of jumping to an
    # arbitrary elbow-up/elbow-down configuration.
    if seed_qpos:
        for name, val in seed_qpos.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{name}")
            data.qpos[model.jnt_qposadr[jid]] = val
        mujoco.mj_forward(model, data)

    goal_pos = data.xpos[target_body_id].copy()

    jacp = np.zeros((3, model.nv))
    n_dof = len(dof_adrs)

    for iteration in range(MAX_ITERS):
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        pos_error = goal_pos - current_pos
        err_norm = float(np.linalg.norm(pos_error))

        if err_norm < POS_TOL_M:
            break

        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # Restrict Jacobian to the arm's own DOFs.
        j_sub = jacp[:, dof_adrs]

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * err
        jjt = j_sub @ j_sub.T
        damped = jjt + (DLS_DAMPING ** 2) * np.eye(3)
        dq = j_sub.T @ np.linalg.solve(damped, pos_error)

        dq = np.clip(dq, -STEP_CLAMP_RAD, STEP_CLAMP_RAD)

        for i, adr in enumerate(qpos_adrs):
            data.qpos[adr] += dq[i]

    mujoco.mj_forward(model, data)
    final_pos = data.site_xpos[site_id].copy()
    final_err = float(np.linalg.norm(goal_pos - final_pos))

    solution = {name: float(data.qpos[adr]) for name, adr in zip(ARM_JOINT_NAMES, qpos_adrs)}

    return {
        "converged": final_err < POS_TOL_M,
        "iterations_used": iteration + 1,
        "final_position_error_m": final_err,
        "goal_position": goal_pos.tolist(),
        "achieved_position": final_pos.tolist(),
        "joint_solution_rad": solution,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/dual_so101_no_table_collision.xml",
        help="Path to the collision-corrected MuJoCo model.",
    )
    parser.add_argument(
        "--seed-lift",
        action="store_true",
        help="Seed IK from the LIFT pose (recommended -- keeps the solution "
             "close to the existing PRE_GRASP/GRASP/LIFT trajectory instead "
             "of jumping to an unrelated arm configuration).",
    )
    parser.add_argument("--out", default="results/place_ik_resolve.json")
    args = parser.parse_args()

    seed = None
    if args.seed_lift:
        seed = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.8,
            "elbow_flex": 1.1,
            "wrist_flex": -0.3,
            "wrist_roll": 0.0,
        }

    result = solve_ik(args.model, seed_qpos=seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("=== PLACE IK RE-SOLVE ===")
    print(f"converged: {result['converged']}  "
          f"(iters={result['iterations_used']}, "
          f"final_error={result['final_position_error_m']:.6f} m)")
    print("\nNew PLACE joint values (radians):")
    for name, val in result["joint_solution_rad"].items():
        print(f"  {name:16s} {val:+.10f}")

    print(f"\nFull result written to: {out_path.resolve()}")

    if not result["converged"]:
        print(
            "\nWARNING: IK did not converge within tolerance. This can mean "
            "place_target is out of reach, or a residual collision is still "
            "blocking the pose -- check the 3 remaining gripper<->table "
            "contacts before trusting this solution."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
