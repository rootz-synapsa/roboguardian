"""
Generic arm IK solver.

Damped least-squares IK, factored out of the earlier PLACE re-solve
script so it can be reused by the recovery controller (regrasp target)
without duplicating the loop.

Solves on a fresh MjData internally -- never touches whatever live
MjData your control loop is running, so calling this mid-episode is
safe and side-effect-free on the real simulation.
"""

from __future__ import annotations

import mujoco
import numpy as np


def solve_arm_ik(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: list[str],
    goal_pos: np.ndarray,
    seed_qpos: dict[str, float] | None = None,
    max_iters: int = 200,
    damping: float = 0.05,
    pos_tol_m: float = 1e-4,
    step_clamp_rad: float = 0.2,
) -> dict:
    """Solve for radian values of `joint_names` (each must have a
    matching `left_<name>` joint) that bring `site_name` to `goal_pos`."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise LookupError(f"site not found: {site_name}")

    qpos_adrs, dof_adrs = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{name}"
        )
        if jid < 0:
            raise LookupError(f"joint not found: left_{name}")
        qpos_adrs.append(model.jnt_qposadr[jid])
        dof_adrs.append(model.jnt_dofadr[jid])

    if seed_qpos:
        for name, val in seed_qpos.items():
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{name}"
            )
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = val
        mujoco.mj_forward(model, data)

    goal_pos = np.asarray(goal_pos, dtype=float)
    jacp = np.zeros((3, model.nv))
    iteration = 0

    for iteration in range(max_iters):
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        pos_error = goal_pos - current_pos

        if float(np.linalg.norm(pos_error)) < pos_tol_m:
            break

        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        j_sub = jacp[:, dof_adrs]
        jjt = j_sub @ j_sub.T
        damped = jjt + (damping ** 2) * np.eye(3)
        dq = j_sub.T @ np.linalg.solve(damped, pos_error)
        dq = np.clip(dq, -step_clamp_rad, step_clamp_rad)

        for i, adr in enumerate(qpos_adrs):
            data.qpos[adr] += dq[i]

    mujoco.mj_forward(model, data)
    final_pos = data.site_xpos[site_id].copy()
    final_err = float(np.linalg.norm(goal_pos - final_pos))

    solution = {
        name: float(data.qpos[adr])
        for name, adr in zip(joint_names, qpos_adrs)
    }

    return {
        "converged": final_err < pos_tol_m,
        "iterations_used": iteration + 1,
        "final_position_error_m": final_err,
        "goal_position": goal_pos.tolist(),
        "joint_solution_rad": solution,
    }
