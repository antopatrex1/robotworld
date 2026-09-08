#!/usr/bin/env python3
"""Export a portable MJCF and model schema for a later GPU integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_world.scene import ROOT, build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="lab")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gpu_bundle")
    args = parser.parse_args()
    model, data, _, xml = build_scene(args.environment, assisted=False)
    folder = args.output.resolve()
    (folder / "assets").mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml)
    for asset in tree.findall("./asset/*"):
        if asset.get("file"):
            source = Path(asset.get("file"))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
            name = f"{digest}_{source.name}"
            shutil.copy2(source, folder / "assets" / name)
            asset.set("file", "assets/" + name)
    tree.write(folder / "scene.xml", encoding="unicode")
    # Verify portable asset resolution, including retained hand tendons.
    portable = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert (portable.nq, portable.nv, portable.nu) == (model.nq, model.nv, model.nu)
    np.savez_compressed(folder / "initial_state.npz", qpos=data.qpos, ctrl=data.ctrl)
    manifest = {
        "environment":args.environment, "nq":model.nq, "nv":model.nv, "nu":model.nu,
        "units":"metres, radians", "quaternion":"wxyz", "up_axis":"Z",
        "joints":[{"name":model.joint(i).name,"type":int(model.jnt_type[i]),"qpos_address":int(model.jnt_qposadr[i]),"dof_address":int(model.jnt_dofadr[i])} for i in range(model.njnt)],
        "actuators":[model.actuator(i).name for i in range(model.nu)],
        "source_revisions":json.loads((ROOT / "sources.lock.json").read_text()),
        "spider_status":"custom composite embodiment adapter and hand/object reference data required; not an out-of-box SPIDER task",
        "base_support_active":False,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (folder / "licenses").mkdir(exist_ok=True)
    for name in ("unitree_g1", "shadow_hand"):
        shutil.copy2(ROOT / "vendor/mujoco_menagerie" / name / "LICENSE", folder / "licenses" / (name + ".txt"))
    print(f"Portable model validated: {folder}")


if __name__ == "__main__":
    main()
