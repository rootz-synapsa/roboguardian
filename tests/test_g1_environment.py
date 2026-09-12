#!/usr/bin/env python3

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    try:
        import mujoco
        import openvino as ov

        xml = """
        <mujoco>
          <worldbody>
            <body name="arm_base" pos="0 0 0">
              <joint name="arm_joint" type="slide" axis="1 0 0"/>
              <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
            </body>

            <body name="target_object" pos="0.5 0 0.05">
              <joint name="object_joint" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.04" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_object"
        )
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_joint"
        )

        initial_pos = data.xpos[body_id].copy()

        core = ov.Core()
        param = ov.opset13.parameter(
            [1, 3, 64, 64],
            dtype=np.float32,
            name="input"
        )
        result = ov.opset13.result(param)
        ov_model = ov.Model([result], [param], "identity_model")
        compiled = core.compile_model(ov_model, "CPU")

        dummy_image = np.random.randn(1, 3, 64, 64).astype(np.float32)
        perception_output = compiled([dummy_image])[0]

        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] += 0.10
        mujoco.mj_forward(model, data)

        displaced_pos = data.xpos[body_id].copy()

        displacement = float(
            np.linalg.norm(displaced_pos - initial_pos)
        )
        detected = displacement > 0.05

        evidence = {
            "gate": "G1a",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mujo_co_version": mujoco.__version__,
            "openvino_version": ov.__version__,
            "generic_mujoco_model_loaded": True,
            "openvino_compiled": True,
            "openvino_inference_shape": list(perception_output.shape),
            "object_pose_readable": True,
            "initial_pose": initial_pos.tolist(),
            "displaced_pose": displaced_pos.tolist(),
            "displacement_meters": displacement,
            "displacement_detected": bool(detected),
            "status": "PASS" if detected else "FAIL",
            "note": (
                "This verifies the base toolchain only. "
                "Dual SO-101 competition environment remains pending."
            )
        }

        Path("results").mkdir(exist_ok=True)

        with open("results/g1_environment.json", "w") as f:
            json.dump(evidence, f, indent=2)

        print(json.dumps(evidence, indent=2))

        if not detected:
            return 1

        print("G1A_BASE_TOOLCHAIN=PASS")
        return 0

    except Exception as e:
        print(f"G1A_BASE_TOOLCHAIN=FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
