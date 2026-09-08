#!/usr/bin/env python3
"""Convert upstream retargeting results into a checked local reference archive."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_world.scene import build_scene
from robot_world.motion import SomaMotion, load_spider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["soma", "spider"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-scene", type=Path)
    parser.add_argument("--environment", default="lab")
    parser.add_argument("--fps", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, data, _, _ = build_scene(args.environment)
    if args.source == "soma":
        motion = SomaMotion(args.input, model, data.qpos, args.fps)
        qpos, times = motion.qpos, motion.time
        report = {"mapped": motion.joint_names, "unmapped": [], "mode": "body reference; fingers need separate retargeting"}
    else:
        if args.source_scene is None:
            parser.error("SPIDER import requires --source-scene to resolve joint order.")
        result = load_spider(args.input, args.source_scene, model, data.qpos)
        qpos = result.pop("qpos")
        times = result.pop("time")
        if times is None:
            parser.error("SPIDER archive must include timestamps; do not guess the control rate.")
        report = result
    if len(times) != len(qpos) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        parser.error("Motion timestamps must be finite, increasing, and match the frame count.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, qpos=qpos, time=times-times[0],
                        joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),
                        source=np.array(args.source), mode=np.array("reference_only"))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames":len(qpos), "duration":float(times[-1]-times[0]), **report}, indent=2))


if __name__ == "__main__":
    main()
