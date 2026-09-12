#!/usr/bin/env python3

import sys


def main():
    try:
        import mujoco

        print(f"MuJoCo version: {mujoco.__version__}")

        xml = """
        <mujoco>
          <worldbody>
            <body name="slider" pos="0 0 0">
              <joint name="slider_joint" type="slide" axis="1 0 0"/>
              <geom type="box" size="0.05 0.05 0.05"/>
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

        for _ in range(10):
            mujoco.mj_step(model, data)

        print(f"nq={model.nq}, nv={model.nv}")
        print("Simulation steps: 10")
        print("G0_MUJOCO=PASS")
        return 0

    except Exception as e:
        print(f"G0_MUJOCO=FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
