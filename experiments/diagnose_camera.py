"""
Camera diagnostic for G7 dataset generation
==============================================

The training images came back nearly identical regardless of plate
position (mean/std diff in the 3rd decimal). This script finds out why
by rendering every named camera in the model at two extreme plate
positions and reporting how much each camera's image actually changes.

A camera that "sees" the workspace should show a large pixel diff
between the two positions. A camera that shows ~0 diff either isn't
pointed at the workspace, or the plate isn't in its frustum at all.

Usage:
    python experiments/diagnose_camera.py --model models/dual_so101_no_table_collision.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def list_cameras(model: mujoco.MjModel) -> list[str]:
    names = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        names.append(name or f"cam_{i}")
    return names


def render_at_offset(model, data, renderer, plate_q_adr, base_pose, dx, dy, camera):
    data.qpos[plate_q_adr:plate_q_adr + 3] = base_pose + np.array([dx, dy, 0.0])
    data.qpos[plate_q_adr + 3:plate_q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    if camera is None:
        renderer.update_scene(data)
    else:
        renderer.update_scene(data, camera=camera)
    return renderer.render()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_no_table_collision.xml")
    parser.add_argument("--img-size", type=int, default=128,
                         help="Larger than training size so saved PNGs are easier to inspect by eye.")
    parser.add_argument("--offset", type=float, default=0.15)
    parser.add_argument("--out-dir", default="results/camera_diagnostics")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    plate_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    if plate_jnt < 0:
        raise LookupError("joint not found: plate_joint")
    q_adr = model.jnt_qposadr[plate_jnt]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base_pose = data.qpos[q_adr:q_adr + 3].copy()

    cameras = list_cameras(model)
    print(f"Named cameras in model: {cameras if cameras else '(none -- only the default free camera exists)'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)

    candidates = [None] + cameras  # None = default free camera
    results = []

    for cam in candidates:
        label = cam if cam is not None else "DEFAULT_FREE_CAMERA"

        frame_a = render_at_offset(model, data, renderer, q_adr, base_pose, -args.offset, 0.0, cam)
        frame_b = render_at_offset(model, data, renderer, q_adr, base_pose, +args.offset, 0.0, cam)

        diff = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
        mean_abs_diff = float(diff.mean())
        max_abs_diff = float(diff.max())

        results.append((label, mean_abs_diff, max_abs_diff))
        print(f"camera={label:24s} mean_abs_diff={mean_abs_diff:6.3f}  max_abs_diff={max_abs_diff:6.1f}")

        if HAVE_PIL:
            Image.fromarray(frame_a).save(out_dir / f"{label}_offset_neg.png")
            Image.fromarray(frame_b).save(out_dir / f"{label}_offset_pos.png")

    renderer.close()

    print("\n=== VERDICT ===")
    usable = [r for r in results if r[1] > 1.0]  # arbitrary but reasonable "actually changed" cutoff
    if usable:
        best = max(usable, key=lambda r: r[1])
        print(f"Best candidate camera: '{best[0]}' (mean_abs_diff={best[1]:.3f})")
        print("Use this camera explicitly in dataset_gen.py / openvino_provider.py:")
        print(f'    renderer.update_scene(data, camera="{best[0]}")' if best[0] != "DEFAULT_FREE_CAMERA"
              else "    renderer.update_scene(data)  # default is actually fine")
    else:
        print("No camera shows meaningful change at this offset. Likely causes:")
        print("  - no camera in the XML is aimed at the workspace at all -- need to add one")
        print("  - the plate is out of frame / occluded for every camera")
        print("  - offset (0.15m) is too small relative to camera framing -- try --offset 0.3 to confirm")
        if HAVE_PIL:
            print(f"\nInspect the saved PNGs in {out_dir.resolve()} to see what each camera is actually looking at.")
        else:
            print("\n(Install Pillow: pip install pillow --break-system-packages -- to save PNGs for visual inspection.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
