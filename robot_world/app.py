"""Native Mac simulation. Launch with mjpython -m robot_world.app."""
import argparse
import json
import queue
import time
from pathlib import Path

import mujoco
import numpy as np

from .scene import ROOT, build_scene
from .control import Controller
from .motion import SomaMotion, ReferenceMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="lab")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--motion", choices=["walk", "pickup"])
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--render", type=Path)
    parser.add_argument("--unassisted", action="store_true")
    parser.add_argument("--reference", type=Path, help="Archive produced by scripts/motion_bridge.py")
    args = parser.parse_args()
    choices = [p.stem for p in sorted((ROOT / "configs/environments").glob("*.json"))]
    pending = queue.SimpleQueue()
    environment = args.environment
    while True:
        model, data, config, _ = build_scene(environment, assisted=not args.unassisted)
        controller = Controller(model, data)
        motion_folder = ROOT / "vendor/soma-retargeter/assets/motions/csv"
        def load_motion(name):
            filename = "Neutral_walk_forward_002__A057.csv" if name == "walk" else "small_light_one_hand_pick_up_front_low_002__A507.csv"
            motion = SomaMotion(motion_folder / filename, model, controller.home_qpos)
            if name == "walk":
                distance = np.linalg.norm(motion.qpos[:, :2], axis=1)
                end = np.flatnonzero(distance > 1.3)
                if len(end):
                    motion.duration = motion.time[end[0]]
                motion.qpos[:, 0] -= 0.6
                motion.qpos[:, 1] -= 1.0
            controller.start_motion(motion)
        if args.motion:
            load_motion(args.motion)
        if args.pick:
            controller.start_pick()
        if args.reference:
            controller.start_motion(ReferenceMotion(args.reference, model), include_hands=True)
        if args.headless:
            for _ in range(int(args.seconds / model.opt.timestep)):
                controller.step()
            print(json.dumps(controller.metrics(), indent=2))
            if args.render:
                from PIL import Image
                with mujoco.Renderer(model, height=900, width=1280) as renderer:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = [0.25, 0, 0.7]
                    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -18
                    renderer.update_scene(data, camera=camera)
                    args.render.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(renderer.render()).save(args.render)
            return
        from mujoco import viewer as mjviewer
        print(f"\n{config['name']}\n1–{len(choices)}: change scene {choices}\nW: SOMA walk   P: SOMA pickup reference   G: physical grasp attempt\nO/C: open/close hands   R: reset   B: toggle base support\nArrow keys: move base-support target   Space: pause\n", flush=True)
        next_environment = None
        paused = False
        with mjviewer.launch_passive(model, data, key_callback=pending.put, show_left_ui=True, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = [0.3, 0, 0.7]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -18
            while viewer.is_running():
                started = time.monotonic()
                while not pending.empty():
                    key = pending.get()
                    if ord('1') <= key < ord('1') + len(choices):
                        next_environment = choices[key - ord('1')]
                    elif key == ord('W'):
                        load_motion("walk")
                    elif key == ord('P'):
                        load_motion("pickup")
                    elif key == ord('G'):
                        controller.start_pick()
                    elif key in (ord('O'), ord('C')):
                        controller.hand_closure = float(key == ord('C'))
                    elif key == ord('R'):
                        controller.reset()
                    elif key == ord('B'):
                        eqid = model.equality("base_support").id
                        data.eq_active[eqid] = not data.eq_active[eqid]
                        controller.status = "Base support ON" if data.eq_active[eqid] else "UNASSISTED PHYSICS · no balance controller"
                    elif key == 32:
                        paused = not paused
                    elif key in (262, 263, 264, 265):
                        controller.motion = None
                        data.mocap_pos[0, 0 if key in (264, 265) else 1] += 0.08 * (1 if key in (263, 265) else -1)
                if next_environment:
                    break
                with viewer.lock():
                    if not paused:
                        for _ in range(8):
                            controller.step()
                    viewer.user_scn.ngeom = 0
                    # A visible support marker makes the assisted mode apparent.
                    if data.eq_active[model.equality("base_support").id]:
                        geom = viewer.user_scn.geoms[0]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.045]*3), data.mocap_pos[0] + [0, 0, 0.8], np.eye(3).reshape(-1), np.array([1., 0.65, 0.1, 0.8]))
                        viewer.user_scn.ngeom = 1
                metrics = controller.metrics()
                support = "ON - assisted" if metrics["base_support_active"] else "OFF - no balance policy"
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     config["name"] + "\nG1 + five-finger Shadow hands\nBase support\nHand/object contacts\nPeak bottle lift",
                     "\n" + controller.status + f"\n{support}\n{metrics['hand_object_contacts']}\n{metrics['peak_bottle_lift_m']:.3f} m"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "Scenes\nW / P\nG\nO / C\nR / B / Space",
                     "  ".join(f"{i+1}:{n}" for i,n in enumerate(choices)) + "\nWalk / pickup reference\nExperimental physical grasp\nOpen / close fingers\nReset / base support / pause")])
                viewer.sync()
                time.sleep(max(0, 0.016 - (time.monotonic() - started)))
        if not next_environment:
            return
        environment = next_environment
        args.motion, args.pick = None, False


if __name__ == "__main__":
    main()
