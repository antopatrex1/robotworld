"""Compose upstream G1 and Shadow MJCF without editing their source models."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from .objects import add_props

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "vendor/mujoco_menagerie"


def available_environments(root=ROOT):
    """List built-in scenes and imported worlds whose manifests are installed."""
    options={}
    for path in sorted((root/'configs/environments').glob('*.json')):
        config=json.loads(path.read_text())
        manifest=config.get('world_manifest')
        if manifest and not (root/manifest).is_file():
            continue
        options[config['name']]=path.stem
    return options


def section(root, name):
    found = root.find(name)
    return ET.SubElement(root, name) if found is None else found


def absolute_assets(root, directory):
    for element in root.findall("./asset/*"):
        if "file" in element.attrib:
            if element.tag == "mesh" and "name" not in element.attrib:
                element.set("name", Path(element.get("file")).stem)
            element.set("file", str((directory / "assets" / element.get("file")).resolve()))


def attach_hand(robot, side):
    hand = ET.parse(MENAGERIE / "shadow_hand" / f"{side}_hand.xml").getroot()
    absolute_assets(hand, MENAGERIE / "shadow_hand")
    # Prefix every named resource and reference, including inferred mesh names.
    prefix = f"{side}_shadow_"
    references = {"name", "class", "childclass", "mesh", "material", "joint", "tendon", "body1", "body2"}
    for element in hand.iter():
        for attr in references.intersection(element.attrib):
            element.set(attr, prefix + element.get(attr))
    palm_name = prefix + ("rh_palm" if side == "right" else "lh_palm")
    palm = copy.deepcopy(hand.find(f".//body[@name='{palm_name}']"))
    # G1 already has a 3-axis wrist: retain the Shadow palm/fingers, omit its
    # bench-mounted forearm and the two redundant wrist joints/actuators.
    for joint in list(palm.findall("joint")):
        palm.remove(joint)
    palm.set("childclass", prefix + f"{side}_hand")
    palm.set("pos", "0.045 0 0")
    palm.set("quat", "0.70710678 0 0.70710678 0")
    for site in list(palm.findall("site")):
        palm.remove(site)
    ET.SubElement(palm, "site", name=f"{side}_grasp", pos="0 -0.035 0.09", size="0.006", rgba="0.1 0.9 0.7 1", group="4")
    wrist = robot.find(f".//body[@name='{side}_wrist_yaw_link']")
    for geom in list(wrist.findall("geom")):
        if "rubber_hand" in geom.get("mesh", ""):
            wrist.remove(geom)
    wrist.append(palm)
    for element in hand.find("default"):
        robot.find("default").append(copy.deepcopy(element))
    for tag in ("asset", "tendon", "actuator", "contact"):
        for element in hand.find(tag):
            if tag == "actuator" and "_WRJ" in element.get("joint", ""):
                continue
            if tag == "contact" and "forearm" in element.get("body2", ""):
                continue
            section(robot, tag).append(copy.deepcopy(element))


def add_furniture(world, name, pos, size, rgba):
    ET.SubElement(world, "geom", name=name, type="box", pos=pos, size=size,
                  rgba=rgba, friction="0.8 0.01 0.001")


def build_scene(environment="lab", assisted=True):
    config_path = ROOT / "configs/environments" / f"{environment}.json"
    config = json.loads(config_path.read_text())
    robot = ET.parse(MENAGERIE / "unitree_g1/g1.xml").getroot()
    robot.set("model", "G1 with bilateral five-finger Shadow hands")
    absolute_assets(robot, MENAGERIE / "unitree_g1")
    robot.find("compiler").attrib.pop("meshdir", None)
    key = np.fromstring(robot.find("./keyframe/key").get("qpos"), sep=" ")
    bare = mujoco.MjModel.from_xml_path(str(MENAGERIE / "unitree_g1/g1.xml"))
    home_joints = {bare.joint(i).name: key[bare.jnt_qposadr[i]] for i in range(1, bare.njnt)}
    robot.remove(robot.find("keyframe"))
    for side in ("left", "right"):
        attach_hand(robot, side)
    robot.find("option").attrib.update(timestep="0.002", cone="elliptic", impratio="10", iterations="80")
    # Flat cylinder bases need a contact manifold to rest stably on the table.
    # A single convex contact caused the untouched mug to creep ~18mm in 10s.
    section(robot.find("option"),"flag").set("multiccd","enable")
    visual = section(robot, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="900")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6", specular="0.2 0.2 0.2")
    asset, world = robot.find("asset"), robot.find("worldbody")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.13 0.19 0.25", rgb2="0.025 0.035 0.06", width="512", height="3072")
    ET.SubElement(asset, "texture", name="floor_grid", type="2d", builtin="checker", rgb1=config["floor_color"], rgb2=config["floor_alt"], width="512", height="512")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_grid", texrepeat="12 12", reflectance="0.08")
    ET.SubElement(world, "geom", name="floor", type="plane", size="12 12 0.1", material="floor_material", friction="0.9 0.01 0.001")
    ET.SubElement(world, "light", pos="0 -2 4", dir="0 0 -1", diffuse="0.7 0.8 0.9")
    ET.SubElement(world, "camera", name="overview", pos="2.3 -2.5 1.7", xyaxes="0.74 0.67 0 -0.22 0.24 0.94")
    for furniture in config.get("furniture", []):
        add_furniture(world, **furniture)
    if config.get("world_manifest"):
        manifest = json.loads((ROOT / config["world_manifest"]).read_text())
        for proxy in manifest.get("collision_boxes", []):
            add_furniture(world, **proxy)
            world[-1].set("group", "3")
        for index, mesh in enumerate(manifest.get("visual_meshes", [])):
            mesh_name = f"world_visual_{index}"
            ET.SubElement(asset, "mesh", name=mesh_name, file=str((ROOT / mesh).resolve()))
            ET.SubElement(world, "geom", name=mesh_name, mesh=mesh_name, type="mesh",
                          rgba="0.62 0.67 0.69 1", contype="0", conaffinity="0", group="1")
    # A physical movable household bottle, separate from generated room geometry.
    add_furniture(world, "work_table", "0.57 -0.1 0.70", "0.34 0.4 0.025", "0.43 0.28 0.16 1")
    for x in (0.26, 0.87):
        for y in (-0.43, 0.23):
            add_furniture(world, f"leg_{x}_{y}", f"{x} {y} 0.3375", "0.018 0.018 0.3375", "0.12 0.15 0.18 1")
    ET.SubElement(world, "site", name="place_target", pos="0.28 -0.02 0.728", type="cylinder", size="0.065 0.002", rgba="0.1 0.9 0.65 0.55", group="4")
    bottle = ET.SubElement(world, "body", name="bottle", pos="0.28 -0.27 0.805")
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    ET.SubElement(bottle, "geom", name="bottle_body", type="cylinder", size="0.028 0.075", mass="0.12", rgba="0.12 0.64 0.79 1", friction="1.2 0.01 0.001", condim="4")
    ET.SubElement(bottle, "geom", name="bottle_cap", type="cylinder", pos="0 0 0.087", size="0.016 0.012", mass="0.01", rgba="0.92 0.95 0.96 1")
    ET.SubElement(bottle, "site", name="bottle_center", size="0.003", rgba="1 0.5 0.1 1")
    add_props(world)
    if config.get('camera_layout'):
        from .table_alignment import apply_camera_layout
        apply_camera_layout(world, asset, ROOT, config)
    ET.SubElement(world, "body", name="support_target", mocap="true", pos="0 0 0.79")
    ET.SubElement(section(robot, "equality"), "weld", name="base_support", body1="support_target", body2="pelvis", relpose="0 0 0 1 0 0 0", solref="0.01 1", active=str(assisted).lower())
    output = ROOT / "build" / f"{environment}.xml"
    output.parent.mkdir(exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.79
    for name, value in home_joints.items():
        data.qpos[model.joint(name).qposadr[0]] = value
        data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)
    return model, data, config, output
