"""
G7 — Synthetic training data generator
=========================================

Ground truth is free in simulation: for each rendered frame, the label
is exactly the plate's (x, y) world position at render time, read
directly from MuJoCo. This sidesteps the "no COCO class for plate"
problem entirely and costs nothing but render time.

z is assumed constant (table height) since the disturbance model in
this project only perturbs x/y.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def capture_cropped(renderer, data, camera, crop_rows, crop_cols, train_size) -> np.ndarray:
    """Render at the renderer's full (large) resolution, crop to the
    workspace ROI found by diagnose_plate_visibility.py, then resize
    down to the training resolution. This is the "digital zoom" fix:
    the plate occupied <1% of the raw frame, which was invisible to a
    tiny CNN's global average pool -- cropping first raises that to
    ~5-6%, per the diagnostic."""
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()  # HxWx3 uint8, at renderer's configured size
    cropped = frame[crop_rows[0]:crop_rows[1], crop_cols[0]:crop_cols[1]]
    resized = np.array(Image.fromarray(cropped).resize((train_size, train_size), Image.BILINEAR))
    return resized


def generate(model_path: str, out_dir: str, n_samples: int,
             x_range: tuple[float, float], y_range: tuple[float, float],
             seed: int, camera: str, render_size: int,
             crop_rows: tuple[int, int], crop_cols: tuple[int, int],
             train_size: int, arm_pose: list[float], arm_pose_jitter_std: float) -> None:
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    if plate_jnt < 0:
        raise LookupError("joint not found: plate_joint")
    q_adr = model.jnt_qposadr[plate_jnt]

    # Match the arm pose to wherever observe() is actually called during
    # the real episode (PRE_GRASP, by default) -- training on the reset
    # pose while inference happens mid-trajectory is exactly the domain
    # gap that caused false_recovery_rate=100% in the vision G4 run.
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    arm_qpos_adrs = []
    for name in arm_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{name}")
        if jid < 0:
            raise LookupError(f"joint not found: left_{name}")
        arm_qpos_adrs.append(model.jnt_qposadr[jid])

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base_pose = data.qpos[q_adr:q_adr + 3].copy()
    z_fixed = float(base_pose[2])

    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    out_path = Path(out_dir)
    (out_path / "images").mkdir(parents=True, exist_ok=True)

    labels = []
    for i in range(n_samples):
        x = base_pose[0] + rng.uniform(*x_range)
        y = base_pose[1] + rng.uniform(*y_range)
        data.qpos[q_adr:q_adr + 3] = [x, y, z_fixed]
        data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Small per-sample jitter around arm_pose so the detector sees a
        # little variation, not one exact frozen configuration -- this
        # buys some robustness against the slightly different arm pose
        # reached after a regrasp during recovery re-observation.
        jitter = (rng.normal(0.0, arm_pose_jitter_std, size=len(arm_pose))
                  if arm_pose_jitter_std > 0 else np.zeros(len(arm_pose)))
        for adr, base_val, j in zip(arm_qpos_adrs, arm_pose, jitter):
            data.qpos[adr] = base_val + j

        mujoco.mj_forward(model, data)

        frame = capture_cropped(renderer, data, camera, crop_rows, crop_cols, train_size)

        fname = f"{i:05d}.npy"
        np.save(out_path / "images" / fname, frame)
        labels.append({"file": fname, "x": float(x), "y": float(y)})

        if (i + 1) % 100 == 0:
            print(f"  rendered {i + 1}/{n_samples}")

    renderer.close()
    (out_path / "labels.json").write_text(json.dumps({
        "z_fixed": z_fixed,
        "img_size": train_size,
        "n_samples": n_samples,
        "camera": camera,
        "render_size": render_size,
        "crop_rows": list(crop_rows),
        "crop_cols": list(crop_cols),
        "arm_pose": list(arm_pose),
        "arm_pose_jitter_std": arm_pose_jitter_std,
        "samples": labels,
    }, indent=2))
    print(f"Wrote {n_samples} samples to {out_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_no_table_collision.xml")
    parser.add_argument("--out", default="results/g7_dataset")
    parser.add_argument("--n", type=int, default=800,
                         help="800 samples trains in a couple minutes on CPU for a model this small.")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.15, 0.15])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.15, 0.15])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera", default="overhead_cam")
    parser.add_argument("--render-size", type=int, default=256,
                         help="Raw render resolution before cropping -- needs to be large "
                              "enough that the crop region below stays meaningful.")
    parser.add_argument("--crop-rows", type=int, nargs=2, default=[176, 256],
                         help="Row range (in render-size pixels) from diagnose_plate_visibility.py.")
    parser.add_argument("--crop-cols", type=int, nargs=2, default=[55, 200],
                         help="Col range (in render-size pixels) from diagnose_plate_visibility.py.")
    parser.add_argument("--train-size", type=int, default=128,
                         help="Final resolution fed to the model, after crop+resize.")
    parser.add_argument("--arm-pose", type=float, nargs=6,
                         default=[0.0, -0.9, 1.2, -0.3, 0.0, 0.0],
                         metavar=("SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW_FLEX",
                                  "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
                         help="Arm joint config to render with -- defaults to PRE_GRASP, "
                              "matching where observe() is actually called at the "
                              "before_grasp boundary. Training on the reset pose instead "
                              "of this caused a 100%% false-recovery-rate domain gap.")
    parser.add_argument("--arm-pose-jitter-std", type=float, default=0.03,
                         help="Per-joint Gaussian jitter (radians) around --arm-pose, for "
                              "some robustness to the slightly different pose reached "
                              "after a regrasp during recovery re-observation.")
    args = parser.parse_args()

    generate(args.model, args.out, args.n, tuple(args.x_range), tuple(args.y_range),
              args.seed, args.camera, args.render_size,
              tuple(args.crop_rows), tuple(args.crop_cols), args.train_size,
              list(args.arm_pose), args.arm_pose_jitter_std)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
