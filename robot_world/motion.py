"""Name-based adapters for SOMA CSV and scene-specific SPIDER trajectories."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class SomaMotion:
    def __init__(self, path, model, home, fps=120.0):
        if fps <= 0:
            raise ValueError("FPS must be positive.")
        path = Path(path)
        with path.open() as handle:
            header = handle.readline().strip().split(",")
        expected = ["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"]
        if header[:7] != expected:
            raise ValueError("Not a SOMA G1 CSV; download LFS contents if this is a pointer.")
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if len(values) < 2 or values.shape[1] != len(header) or not np.isfinite(values).all():
            raise ValueError("Invalid motion values or CSV shape.")
        if np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("Frame indices must increase strictly.")
        self.time = (values[:, 0] - values[0, 0]) / fps
        self.duration = self.time[-1]
        self.home = home.copy()
        self.qpos = np.repeat(home[None, :], len(values), axis=0)
        self.qpos[:, :3] = values[:, 1:4] * 0.01  # upstream centimetres
        rotations = Rotation.from_euler("xyz", values[:, 4:7], degrees=True)
        # Rotate the reference to face +X initially, preserving pitch and roll.
        heading = rotations[0].as_euler("xyz")[2]
        align = Rotation.from_euler("z", -heading)
        origin = self.qpos[0, :3].copy()
        origin[2] = 0
        self.qpos[:, :3] = align.apply(self.qpos[:, :3] - origin)
        rotations = align * rotations
        self.slerp = Slerp(self.time, rotations)
        self.qpos[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]  # MuJoCo wxyz
        names = [name.removesuffix("_dof") for name in header[7:]]
        if len(names) != 29 or len(set(names)) != 29:
            raise ValueError("Expected 29 unique G1 body joint columns.")
        for col, name in enumerate(names, 7):
            joint = model.joint(name)
            self.qpos[:, joint.qposadr[0]] = np.deg2rad(values[:, col])
        self.joint_names = names

    def sample(self, seconds):
        t = float(np.clip(seconds, 0, self.duration))
        i = min(np.searchsorted(self.time, t, side="right") - 1, len(self.time) - 2)
        alpha = (t - self.time[i]) / (self.time[i + 1] - self.time[i])
        pose = (1 - alpha) * self.qpos[i] + alpha * self.qpos[i + 1]
        pose[3:7] = self.slerp(t).as_quat()[[3, 0, 1, 2]]
        return pose


def load_spider(path, source_xml, target_model, target_home, world_index=0):
    """Import qpos by source MJCF names. Never reinterpret joint arrays by length.

    SPIDER outputs can be [T,N,nq] or [T,nq]. Controls are not transferred:
    optimised controls depend on the original actuators, physics, and contacts.
    """
    source = mujoco.MjModel.from_xml_path(str(source_xml))
    with np.load(path, allow_pickle=False) as archive:
        qpos = archive["qpos"]
        if qpos.ndim == 3:
            qpos = qpos[:, world_index, :]
        if qpos.ndim != 2 or qpos.shape[1] != source.nq or not np.isfinite(qpos).all():
            raise ValueError("SPIDER qpos does not match its source MJCF.")
        times = archive["time"] if "time" in archive else None
        if times is not None and times.ndim >= 2:
            times = times[:, world_index]
        times = None if times is None else times.reshape(-1)
    output = np.repeat(target_home[None, :], len(qpos), axis=0)
    mapped, missing = [], []
    for j in range(source.njnt):
        name = source.joint(j).name
        target_name = name
        if name.startswith("rh_"):
            target_name = "right_shadow_" + name
        elif name.startswith("lh_"):
            target_name = "left_shadow_" + name
        try:
            target = target_model.joint(target_name)
        except KeyError:
            missing.append(name)
            continue
        if int(source.jnt_type[j]) != int(target.type[0]):
            raise ValueError(f"Joint type mismatch for {name}.")
        width = {0: 7, 1: 4, 2: 1, 3: 1}[int(source.jnt_type[j])]
        src, dst = int(source.jnt_qposadr[j]), int(target.qposadr[0])
        output[:, dst:dst + width] = qpos[:, src:src + width]
        mapped.append(target_name)
    if not mapped:
        raise ValueError("No joint names match the target robot.")
    return {"qpos": output, "time": times, "mapped": mapped, "unmapped": missing,
            "mode": "reference_only; physics must be revalidated for the composite robot"}


class ReferenceMotion:
    """Read a motion_bridge archive for assisted body and finger reference tracking."""
    def __init__(self, path, model):
        with np.load(path, allow_pickle=False) as archive:
            self.qpos = archive["qpos"].copy()
            self.time = archive["time"].copy()
            names = archive["joint_names"].tolist()
        expected = [model.joint(i).name for i in range(model.njnt)]
        if names != expected or self.qpos.shape != (len(self.time), model.nq):
            raise ValueError("Reference joint schema does not match this robot.")
        if len(self.time) < 2 or not np.isfinite(self.qpos).all() or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError("Invalid reference data or timestamps.")
        self.time -= self.time[0]
        self.duration = self.time[-1]
        self.slerp = Slerp(self.time, Rotation.from_quat(self.qpos[:, [4, 5, 6, 3]]))

    sample = SomaMotion.sample
