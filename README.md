# Robot World — Mac prototype

A local interactive MuJoCo simulation with the **exact [Warm and inviting living room](https://marble.worldlabs.ai/world/7f964e6d-dbdd-4cf7-a8fb-9292001f69ba)** requested by the user, a Unitree G1 with two five-finger Shadow hands, and ten labeled household objects on a physical table.

Open **Start Robot World + RealSense.command** to launch the simulation and camera together. Enter your Mac password in Terminal for the camera helper. It reuses this checkout's Robot World server when already running and restarts the camera helper. Keep both Terminal sessions open; closing the RealSense window ends its camera helper. From a shell, run `./"Start Robot World + RealSense.command"` (add `--demo` for synthetic camera frames).

For the simulation alone, open **Start Robot World.command**, or run:

```sh
.venv/bin/python scripts/patch_viser.py
.venv/bin/python -m robot_world.web_app --open-browser
```

The viewer runs at **http://127.0.0.1:8765**. It stays local to this Mac and needs the Python process to remain running. No World Labs API key is sent to the browser. Local playback and commands do not spend API credits.

## RealSense RGB-D camera

The [realsense](realsense/README.md) folder contains the native RGB-D camera window and MCP capture server imported from `oai6`. Open **realsense/Start RealSense.command**, or run `./realsense/scripts/start.sh`. Use `--demo` for a synthetic preview. Robot World's **RealSense · live camera** panel shares that helper and displays color, depth, and center distance alongside the simulation. Preview updates run independently at up to 10 Hz, reconnect automatically, and hide stale images when the camera disconnects. Uncheck **Show live camera** to pause the preview. Synthetic frames are labeled DEMO.

The physical camera view is not registered to the simulated room, and depth is not aligned to color. The native viewer and MCP tools (`camera_status`, `capture_rgbd`) remain available. For a helper with a custom socket, launch Robot World with `--realsense-socket /path/to/camera.sock`.

### Paper-aligned table snapshot

Click **Acquire Scene** at the top of the camera pane to capture a fresh RGB-D frame, locate the reference paper, identify tabletop objects, and replace the simulated table layout. Progress appears below the button. The original scene keeps running during analysis; applying the result resets the simulated robot. Failed capture, missing paper, ambiguous placement, and an environment change during analysis preserve the previous scene. Repeated clicks are ignored while acquisition is busy.

Analysis runs locally using [Ultralytics YOLO11 segmentation](https://docs.ultralytics.com/models/yolo11/) and OpenCV. No cloud vision API is used. The model recognizes common object categories; low-confidence supported detections become generically labeled approximate shapes. Cropped and off-table items are omitted, and unsupported cables or small cards may not be detected. Mug, bottle, apple, and mouse instances support simulated pick commands using their displayed names. Other objects receive approximate physical representations, with grasping unavailable until an appropriate grasp model exists. Mice use an approximate 110 × 66 × 46 mm body and a dedicated low grasp stance with opposing thumb/finger contacts. Saved mouse detections from earlier acquisitions are upgraded automatically; cell phones remain unpickable. Dimensions and masses remain assumed; raw unaligned depth is saved but not used to assign RGB object coordinates.

For a fresh installation, install the analysis dependencies and pinned model once:

```sh
.venv/bin/python -m pip install -r requirements-scene.txt
.venv/bin/python scripts/setup_scene_analysis.py
```

The model download is approximately 20 MB and verified by SHA-256. Ultralytics code and model licensing is AGPL-3.0 (see its licensing terms for other options). Captures, analyzed layouts, and generated models stay in ignored `realsense/captures/`; downloaded weights live in ignored `data/models/`. The last successful acquisition is remembered locally, so returning to the camera-table scene restores it. Paper corners are matched against the last successful capture with forward/backward consistency and sheet-appearance checks, falling back to fresh contour detection when needed. Keep the paper in the same physical table corner with the same orientation; substantial camera movement needs a new calibration reference.

Select **Camera table · paper alignment** from the environment menu, or launch with `--environment camera_table`. Saved acquisitions using the original reflected X mapping are corrected automatically when loaded. Before the first acquisition, it loads the original manually mapped snapshot. It maps the user's 210 × 147 mm paper to the virtual tabletop corner at `(0.23, -0.50, 0.725)` metres: the 210 mm edge runs along +Y, and the 147 mm edge runs along +X toward the camera. In that original snapshot, the black mug's base is approximately 0.33 m along the edge and 0.10 m inward. The original ten demo objects are absent from this scene; return to **Warm and inviting living room** for the standard demo.

[Calibration annotations](configs/calibration/paper_table.json) record the four paper corners and manually selected tabletop contact points from a saved 640×480 RGB frame. A planar homography maps those points to tabletop metres. These annotations describe the original snapshot. Acquire Scene redetects the paper and objects; it does not perform full camera-pose calibration or depth-to-color alignment. The raw capture stays in ignored `realsense/captures/`.

The mug is a physical simulation body using assumed cup dimensions and mass. In the original snapshot, the paper, card, connector, and visible cable are visual proxies without collision response; the cropped container and partial blue sheet are omitted. The mug handle orientation is illustrative. Use `pick up the Observed black mug` (or `pick up the black mug`) to grasp the simulated mug. Commands use its current simulated pose; they do not operate a physical robot or validate the real-world calibration. Navigation and scene reset remain available.

The paper's shallow angle, curl, torn edge, and uncorrected lens distortion limit accuracy. A sensitivity check with paper corners perturbed by ±2 pixels gave mug coordinates of roughly 0.30–0.36 m along the edge and 0.09–0.12 m inward (5th–95th percentiles). This is sensitivity to annotation choices, **not** measured positioning accuracy. Check independent physical distances before using the layout for targeting.

## Prompt controls

Type in **Ask the robot**, then press **Enter** or click **Run command**. Replies below the box report progress or explain why a request cannot run. Supported local language commands include:

| Prompt | Behavior |
|---|---|
| `walk around the table` | Plan and follow a clear route around all four sides |
| `walk backward 1 meter` | Move relative to the robot's heading |
| `walk to x 1.5 y 0.8` | Navigate to a floor coordinate in metres |
| `turn left` / `turn right 45 degrees` | Turn in place |
| `go to the red mug` | Walk to a stance near a named object and face it |
| `reach for the blue bottle` | Move the right arm toward the named object |
| `pick up the blue bottle` | Grasp the bottle, lift it 18 cm, and hold it with finger contacts |
| `grab the red mug` | Walk to the mug, adjust to its current position, then grasp, lift and hold it |
| `grab the green apple` | Walk to the apple, oppose the thumb and fingers, lift about 17 cm and hold |
| `open hands` / `close hands` | Actuate both five-finger hands |
| `stop` / `reset` | Stop motion or restore the robot and all objects |

Polite requests such as `Can you grab the red mug?` also work. The parser accepts one supported action at a time. Unknown requests display supported examples. Out-of-room or occupied destinations are rejected. Navigation uses a conservative 2D footprint; it does not certify full-body collision clearance.

The environment menu switches scenes without relaunching the viewer. **Object labels**, **XYZ axes**, **Floor grid**, and **World Labs room** can be toggled. Camera buttons focus the robot, table, or room. Drag to orbit, right-drag to pan, and scroll to zoom.

The table contains a blue bottle, red mug, blue cube, green apple, yellow ball, purple can, orange bowl, red book, yellow sponge, and TV remote. Each is an independently simulated free body, with labels linked to its live position.

## What runs now

- Real G1 body meshes, mass/inertia, joints and actuators from MuJoCo Menagerie, with bilateral Shadow palms and five articulated fingers per hand. The redundant standalone Shadow wrists/forearms are omitted. The composite has 29 body joints, 44 finger joints, 65 actuators and eight coupled tendons. This is a custom simulation embodiment.
- The requested room's **150,000 Gaussian splats**, including its original appearance. Its public export has no collider mesh; collision columns are estimated from splat occupancy. OpenCV Y-down coordinates are converted to MuJoCo Z-up using the scene's metric scale and ground offset. See [World Labs export specifications](https://docs.worldlabs.ai/marble/export/specs).
- Prompt-driven walking uses real precomputed SOMA leg poses and an explicit base-support constraint. It is **assisted locomotion**, not validated free-standing balance. Turns and path following run locally on CPU.
- Physical props, finger actuation, arm inverse kinematics and a verified blue-bottle side grasp. It starts from standing, lifts approximately 17.4 cm, and has held for 40 seconds in a contact-only test. The thumb opposes the fingers; a clearance path avoids the tabletop. Bounded actuator feedforward and three static-friction postprocessing iterations maintain the grip. Objects are never welded to the hand, teleported, or given external support forces. General object manipulation and placement remain experimental.
- A red-mug command carries through supported walking, observation of the mug's current position, stance adjustment, and a cup-specific grasp. The complete sequence from the starting position lifted the mug approximately 15 cm with contacts across the thumb and fingers. A regression test requires five seconds of sustained hold without table support or external object forces. Multiple convex contacts keep the mug's flat base stable while waiting for a command. Enter and Run use the same ordered submission handler.
- SOMA CSV and SPIDER trajectory conversion by joint names, plus a portable MuJoCo model export. Fresh SOMA retargeting and SPIDER optimization have not run on this Mac. Their NVIDIA workflow and required composite-embodiment work are documented in [PIPELINE.md](PIPELINE.md).

The two earlier private World Labs drafts used 460 credits total. This requested public scene was imported without generating another world. The other generated-room options show untextured collider geometry and retain approximate, uncalibrated dimensions.

## Checks and motion bridges

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/patch_viser.py --check
.venv/bin/python -m robot_world.app --headless --seconds 3
.venv/bin/python scripts/motion_bridge.py soma vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv --output build/walk_reference.npz
.venv/bin/python scripts/export_scene.py --environment warm_living_room
```

The tests cover physical model composition, named motion mapping, unit conversion, real SPZ decoding, exact scene identity, object settling, collision-aware routes and assisted walking. Grasp regression checks include a sustained lift above 15 cm, opposing thumb/finger contacts, small object-position variations, zero external forces, existing force limits, and reset behavior. Browser checks cover prompt execution, readable object labels, the table tour, named-object reaching, grasping, and scene switching.

For an actual SPIDER output, supply its exact source MJCF. Imported trajectories remain references; optimized controls do not transfer automatically to a changed robot:

```sh
.venv/bin/python scripts/motion_bridge.py spider /path/to/trajectory_mjwp.npz --source-scene /path/to/source_scene.xml --output build/spider_reference.npz
mjpython -m robot_world.app --reference build/spider_reference.npz
```

The native MuJoCo viewer remains available with `mjpython -m robot_world.app`; the browser viewer uses ordinary Python and does not require a native graphics window.

To rebuild the requested scene manifest from the downloaded public data:

```sh
.venv/bin/python scripts/import_public_scene.py
```

To import a generated world with a collider mesh:

```sh
.venv/bin/python scripts/import_world.py data/worlds/new_operation.json --name new_room
.venv/bin/python -m robot_world.web_app --environment new_room
```

`--scale`, `--yaw`, and `--floor-height` control collider alignment. Verify them before making physics claims. Generating a new World Labs world is a separate paid API action; the key is stored in ignored, owner-readable `.env.worldlabs`.

## Dependencies and assets

Use Python 3.10–3.12, Git, and Node.js. From a fresh clone:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/bootstrap_assets.py
.venv/bin/python scripts/patch_viser.py
.venv/bin/python -m robot_world.web_app --open-browser
```

The bootstrap fetches pinned G1/Shadow assets, the SOMA walking sample, and the public room. It requires no API key and does not generate a paid world. Use `--check` to verify existing assets offline or `--lab-only` to omit the room, then launch with `--environment lab`. Existing modified assets are preserved.

This requires Node.js on `PATH` to verify the browser message ordering. It applies a version-checked fix to Viser 1.1.0's installed source and bundled client: text updates arrive before Run or Enter, preserving the complete command. It does not download packages. The patch is safe to rerun, and **Start Robot World.command** applies it automatically. Restart the server and reload the viewer after changing the client bundle. `--check` verifies the patch without modifying files.

`sources.lock.json` records the inspected source revisions. Large vendor assets, downloaded worlds, credentials and build outputs are ignored by Git, and are installed separately with the bootstrap script.

G1 is BSD-3-Clause; Shadow and SOMA are Apache-2.0. Model licenses are copied into the portable GPU bundle. SPIDER is **CC BY-NC 4.0**, so its later commercial use requires attention to that license. OpenMind is an optional demonstration-capture route, not an active dependency of this local demo.
