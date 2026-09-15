from pathlib import Path
import xml.etree.ElementTree as ET


MODEL_PATHS = [
    Path("models/dual_so101.xml"),
    Path("models/dual_so101_g3.xml"),
    Path("models/dual_so101_no_table_collision.xml"),
]


def test_so101_model_mesh_assets_are_committed():
    missing = []

    for model_path in MODEL_PATHS:
        root = ET.fromstring(model_path.read_text())
        compiler = root.find("compiler")

        assert compiler is not None

        meshdir = compiler.attrib["meshdir"]
        assets_dir = (model_path.parent / meshdir).resolve()

        for mesh in root.findall("./asset/mesh"):
            mesh_path = assets_dir / mesh.attrib["file"]
            if not mesh_path.exists():
                missing.append(str(mesh_path.relative_to(Path.cwd())))

    assert not missing, f"missing MuJoCo mesh assets: {missing}"
