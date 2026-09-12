#!/usr/bin/env bash
set -e

echo "=== STEP 1: Repair directory structure ==="
mkdir -p tests results docs

echo "=== STEP 2: Rename branch to main ==="
git branch -m main || true

echo "=== STEP 3: Fix .gitignore so gate evidence is tracked ==="
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
.venv/
*.log

# Keep gate evidence in git.
# Ignore transient/raw artifacts later only if needed.
EOF

echo "=== STEP 4: Reset unsupported gate claims ==="
cat > docs/G0_CHECK.md << 'EOF'
# G0: Challenge Sanity Check

**Status:** IN_VALIDATION

Installed toolchain observed:

- MuJoCo 3.13.0
- OpenVINO 2026.3.1
- NumPy 2.5.3

G0 PASS requires actual smoke-test execution with exit code 0 and preserved evidence.

EOF

cat > WORKSPACE_STATE.yaml << 'EOF'
workspace: RoboGuardian
phase: G1 Environment Verification
status: IN_VALIDATION
previous_gate: G0 (PARTIAL)
current_gate: G1
next_gate: G2 Happy-Path Baseline
authorization: HOLD_UNTIL_G1_PASS
EOF

echo "=== STEP 5: Create OpenVINO smoke test ==="
cat > tests/test_openvino_smoke.py << 'PYEOF'
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
PYEOF

echo "=== STEP 6: Create MuJoCo smoke test ==="
cat > tests/test_mujoco_smoke.py << 'PYEOF'
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
PYEOF

echo "=== STEP 7: Create G1 integration test ==="
cat > tests/test_g1_environment.py << 'PYEOF'
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
PYEOF

echo "=== STEP 8: Verify files exist before running ==="
ls -lh tests/
ls -lh docs/
ls -lh results/

echo "=== STEP 9: Run OpenVINO smoke test ==="
python tests/test_openvino_smoke.py
echo "OpenVINO exit=$?"

echo "=== STEP 10: Run MuJoCo smoke test ==="
python tests/test_mujoco_smoke.py
echo "MuJoCo exit=$?"

echo "=== STEP 11: Run G1 base integration ==="
python tests/test_g1_environment.py
echo "G1 exit=$?"

echo "=== STEP 12: Verify evidence artifact ==="
test -f results/g1_environment.json
python -m json.tool results/g1_environment.json

echo "=== STEP 13: Write factual gate summary ==="
cat > results/G1_SUMMARY.md << 'EOF'
# G1 Environment Verification Summary

## Current status

- G0 OpenVINO smoke test: PASS
- G0 MuJoCo smoke test: PASS
- G1a Base Toolchain Integration: PASS
- G1b Dual SO-101 Competition Environment: PENDING

## Verified

- OpenVINO imports
- OpenVINO model compiles on CPU
- OpenVINO inference executes
- MuJoCo imports
- MuJoCo XML model loads
- MuJoCo simulation advances
- Object pose can be read
- Object can be displaced programmatically
- Python detects displacement
- Machine-readable gate evidence is preserved

## Not yet verified

- Dual SO-101 MuJoCo model
- Two-arm actuation
- Competition camera configuration
- Table-setting object scene
- Actual competition manipulation environment

## Gate decision

**G1a PASS**

**G1b PENDING**

**G2 NOT AUTHORIZED until G1b passes.**
EOF

cat > docs/G0_CHECK.md << 'EOF'
# G0: Challenge Sanity Check

**Status:** PASS

Verified by executed smoke tests:

- OpenVINO imports successfully
- OpenVINO model compiles on CPU
- OpenVINO inference executes
- MuJoCo imports successfully
- MuJoCo XML model loads
- MuJoCo simulation steps successfully

See:

- `tests/test_openvino_smoke.py`
- `tests/test_mujoco_smoke.py`
- `results/g1_environment.json`

EOF

cat > WORKSPACE_STATE.yaml << 'EOF'
workspace: RoboGuardian
phase: G1b Competition Environment
status: IN_VALIDATION
previous_gate: G1a Base Toolchain (PASS)
current_gate: G1b Dual SO-101 Competition Environment
next_gate: G2 Happy-Path Baseline
authorization: HOLD_UNTIL_G1B_PASS
EOF

echo "=== STEP 14: Final artifact check ==="
ls -lh \
  tests/test_openvino_smoke.py \
  tests/test_mujoco_smoke.py \
  tests/test_g1_environment.py \
  results/g1_environment.json \
  results/G1_SUMMARY.md \
  docs/G0_CHECK.md \
  WORKSPACE_STATE.yaml

echo "=== STEP 15: Git status ==="
git status --short

echo "=== STEP 16: Commit verified repair ==="
git add \
  .gitignore \
  tests/ \
  results/g1_environment.json \
  results/G1_SUMMARY.md \
  docs/G0_CHECK.md \
  WORKSPACE_STATE.yaml \
  requirements.txt

git commit -m "test: repair and verify G0/G1 base toolchain"

echo "=== DONE ==="
git log --oneline -n 5
