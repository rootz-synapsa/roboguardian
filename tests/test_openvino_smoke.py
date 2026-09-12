#!/usr/bin/env python3

import sys
import numpy as np


def main():
    try:
        import openvino as ov

        print(f"OpenVINO version: {ov.__version__}")

        core = ov.Core()

        param = ov.opset13.parameter(
            [1, 3, 64, 64],
            dtype=np.float32,
            name="input"
        )
        result = ov.opset13.result(param)
        model = ov.Model([result], [param], "identity_model")

        compiled = core.compile_model(model, "CPU")

        input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)
        output = compiled([input_data])[0]

        print(f"Compiled device: CPU")
        print(f"Inference output shape: {output.shape}")
        print("G0_OPENVINO=PASS")
        return 0

    except Exception as e:
        print(f"G0_OPENVINO=FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
