"""
Plate visibility diagnostic (segmentation-based)
====================================================

std=0.55 across samples (even with overhead_cam) suggests the plate
occupies very few pixels in the frame. Instead of guessing camera
parameters, this uses MuJoCo's segmentation rendering to find the
EXACT pixel bounding box of the plate at each of several workspace
positions, and reports:

  - how many pixels the plate actually occupies (confirms/denies the
    "too small" hypothesis with a real number instead of a guess)
  - the union bounding box across the full x/y range, which is exactly
    the crop region you need to "digitally zoom" before training

If your MuJoCo python bindings version doesn't expose
Renderer.enable_segmentation() the same way, the error message will
say so explicitly rather than silently producing garbage.

Usage:
    python experiments/diagnose_plate_visibility.py \\
        --model models/dual_so101_no_table_collision.xml --camera overhead_cam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def get_plate_geom_ids(model: mujoco.MjModel) -> list[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    if body_id < 0:
        raise LookupError("body not found: plate")
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]


def plate_bbox_from_segmentation(seg: np.ndarray, plate_geom_ids: set[int]) -> tuple | None:
    """seg has shape (H, W, 2): channel 0 = object id, channel 1 = object type
    (mjOBJ_GEOM == 5). Returns (row_min, row_max, col_min, col_max, n_pixels)
    or None if the plate isn't visible in this frame at all."""
    obj_id = seg[:, :, 0]
    obj_type = seg[:, :, 1]
    mask = (obj_type == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(obj_id, list(plate_geom_ids))

    if not mask.any():
        return None

    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()), int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dual_so101_no_table_collision.xml")
    parser.add_argument("--camera", default="overhead_cam")
    parser.add_argument("--img-size", type=int, default=256,
                         help="Render larger than training size for a finer-grained bbox estimate.")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.15, 0.15])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.15, 0.15])
    parser.add_argument("--n-samples", type=int, default=25)
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

    plate_geom_ids = set(get_plate_geom_ids(model))
    print(f"plate geom ids: {plate_geom_ids}")

    renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)
    try:
        renderer.enable_segmentation_rendering()
    except AttributeError as exc:
        raise RuntimeError(
            "This MuJoCo python binding version doesn't expose a segmentation "
            "rendering method under either name this script tried "
            "(enable_segmentation / enable_segmentation_rendering). "
            "Check `python -c \"import mujoco; print(mujoco.__version__)\"` and "
            "`python -c \"import mujoco; print([m for m in dir(mujoco.Renderer) if 'seg' in m.lower()])\"` "
            "to find the correct method name for your installed version."
        ) from exc

    xs = np.linspace(args.x_range[0], args.x_range[1], 5)
    ys = np.linspace(args.y_range[0], args.y_range[1], 5)

    all_boxes = []
    n_invisible = 0
    n_total = 0

    for dx in xs:
        for dy in ys:
            n_total += 1
            data.qpos[q_adr:q_adr + 3] = base_pose + np.array([dx, dy, 0.0])
            data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=args.camera)
            seg = renderer.render()

            bbox = plate_bbox_from_segmentation(seg, plate_geom_ids)
            if bbox is None:
                n_invisible += 1
                continue
            all_boxes.append(bbox)

    renderer.close()

    print(f"\nplate visible in {n_total - n_invisible}/{n_total} sampled positions")

    if not all_boxes:
        print("Plate was NEVER visible via segmentation at this camera across the "
              "sampled workspace range. The camera is not aimed at the workspace, "
              "or the workspace range is outside its frustum entirely.")
        return 1

    row_mins, row_maxs, col_mins, col_maxs, areas = zip(*all_boxes)
    area_frac = np.mean(areas) / (args.img_size * args.img_size)

    print(f"mean plate pixel area: {np.mean(areas):.1f} px  "
          f"({area_frac:.2%} of the {args.img_size}x{args.img_size} frame)")

    margin = 0.15  # 15% padding around the union bbox
    row_min, row_max = min(row_mins), max(row_maxs)
    col_min, col_max = min(col_mins), max(col_maxs)
    row_pad = int((row_max - row_min) * margin) + 1
    col_pad = int((col_max - col_min) * margin) + 1

    crop_row_min = max(0, row_min - row_pad)
    crop_row_max = min(args.img_size, row_max + row_pad)
    crop_col_min = max(0, col_min - col_pad)
    crop_col_max = min(args.img_size, col_max + col_pad)

    print(f"\nunion bounding box across workspace (at render size {args.img_size}):")
    print(f"  rows [{row_min}, {row_max}]  cols [{col_min}, {col_max}]")
    print(f"\nsuggested crop region (with {margin:.0%} margin), "
          f"in terms of a {args.img_size}x{args.img_size} render:")
    print(f"  rows [{crop_row_min}, {crop_row_max}]  cols [{crop_col_min}, {crop_col_max}]")
    print(f"  crop size: {crop_row_max - crop_row_min} x {crop_col_max - crop_col_min} px")

    if area_frac < 0.005:
        print("\nVERDICT: plate occupies well under 1% of the frame on average -- "
              "this is almost certainly why the detector couldn't learn anything. "
              "Apply the crop region above before resizing to your training "
              "resolution (64x64 or 128x128), in both dataset_gen.py and "
              "openvino_provider.py.")
    else:
        print("\nVERDICT: plate area doesn't look pathologically small -- if training "
              "still fails after applying the crop, look at contrast/color rather "
              "than size (plate may blend into the table color).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
