#!/usr/bin/env python3
"""Download an already-generated World Labs room and prepare MuJoCo visuals."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_world.scene import ROOT


def download(url, target):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "cdn.marble.worldlabs.ai":
        raise ValueError("Only World Labs CDN asset URLs are accepted.")
    with urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())


def import_world(operation_path, name, scale=1.0, yaw=0.0, floor_height=None):
    if not name.replace("_", "").isalnum() or scale <= 0:
        raise ValueError("Use an alphanumeric environment name and a positive scale.")
    result = json.loads(Path(operation_path).read_text())
    if "done" in result:
        if not result["done"] or result.get("error"):
            raise ValueError("The world operation is not successfully completed.")
        result = result["response"]
    folder = ROOT / "data/worlds" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "world.json").write_text(json.dumps(result, indent=2))
    glb = folder / "collider.glb"
    if not glb.exists():
        download(result["assets"]["mesh"]["collider_mesh_url"], glb)
    scene = trimesh.load(glb, force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    # Marble export coordinates are OpenCV Y-down (see World Labs export specs).
    # Apply the same transform to visual and collision geometry.
    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    transform = trimesh.transformations.rotation_matrix(np.deg2rad(yaw), [0, 0, 1]) @ transform
    transform[:3, :3] *= scale
    mesh.apply_transform(transform)
    if floor_height is None:
        # Estimate a low horizontal surface; record this as uncalibrated.
        horizontal = np.abs(mesh.face_normals[:, 2]) > 0.9
        heights = mesh.triangles_center[horizontal, 2]
        weights = mesh.area_faces[horizontal]
        if len(heights) == 0:
            raise ValueError("Cannot infer floor; supply --floor-height in transformed metres.")
        counts, edges = np.histogram(heights, bins=100, weights=weights)
        candidates = np.flatnonzero(counts >= counts.max() * 0.2)
        floor = float((edges[candidates[0]] + edges[candidates[0] + 1]) / 2)
    else:
        floor = floor_height
    mesh.apply_translation([0, 0, -floor])
    transform[2, 3] -= floor
    visual = folder / "room.obj"
    mesh.export(visual)
    # Conservative 20 cm surface-column proxies preserve the room opening.
    # A single convex collision mesh would incorrectly fill the entire room.
    cell = 0.2
    samples = mesh.triangles_center
    samples = samples[(samples[:, 2] > 0.12) & (samples[:, 2] < 1.45)]
    bins = {}
    for x, y, z in samples:
        grid = (int(np.floor(x / cell)), int(np.floor(y / cell)))
        count, height = bins.get(grid, (0, 0.0))
        bins[grid] = (count + 1, max(height, float(z)))
    proxies = []
    for (i, j), (count, height) in sorted(bins.items()):
        if count < 3:
            continue
        proxies.append({"name": f"room_proxy_{i}_{j}", "pos": f"{(i+.5)*cell} {(j+.5)*cell} {height/2}",
                        "size": f"{cell/2} {cell/2} {height/2}", "rgba": "0.2 0.7 0.9 0"})
    manifest = {
        "world_id": result["world_id"], "marble_url": result.get("world_marble_url"),
        "visual_meshes": [str(visual.relative_to(ROOT))],
        "world_to_sim": transform.tolist(), "source_up": "-Y", "sim_up": "Z",
        "units": "metres", "scale": scale, "floor_estimate": floor,
        "calibration": "automatic estimate; verify scale and floor before physics evaluation",
        "collision_mode": "conservative 20 cm height-column proxies; no underpasses or overhangs",
        "collision_boxes": proxies,
        "splat_urls": result["assets"].get("splats", {}).get("spz_urls", {}),
        "bounds": mesh.bounds.tolist(), "triangles": len(mesh.faces),
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    config = {"name": f"World Labs · {name.replace('_', ' ').title()}",
              "floor_color": "0.31 0.27 0.23", "floor_alt": "0.34 0.3 0.26",
              "world_manifest": str(manifest_path.relative_to(ROOT)), "furniture": []}
    (ROOT / "configs/environments" / f"{name}.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"environment": name, "triangles": len(mesh.faces), "floor_estimate": floor, "bounds": mesh.bounds.tolist()}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--floor-height", type=float)
    args = parser.parse_args()
    import_world(args.operation, args.name, args.scale, args.yaw, args.floor_height)
