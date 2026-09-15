from pathlib import Path
import xml.etree.ElementTree as ET


def test_g3_model_mesh_assets_are_committed():
    model_path = Path("models/dual_so101_g3.xml")
    root = ET.fromstring(model_path.read_text())
    compiler = root.find("compiler")

    assert compiler is not None

    meshdir = compiler.attrib["meshdir"]
    assets_dir = (model_path.parent / meshdir).resolve()

    missing = []
    for mesh in root.findall("./asset/mesh"):
        mesh_path = assets_dir / mesh.attrib["file"]
        if not mesh_path.exists():
            missing.append(str(mesh_path.relative_to(Path.cwd())))

    assert not missing, f"missing MuJoCo mesh assets: {missing}"
