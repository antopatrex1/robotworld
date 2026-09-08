# Robot World — Mac prototype

A local interactive MuJoCo simulation with the **exact [Warm and inviting living room](https://marble.worldlabs.ai/world/7f964e6d-dbdd-4cf7-a8fb-9292001f69ba)** requested by the user, a Unitree G1 with two five-finger Shadow hands, and ten labeled household objects on a physical table.

Open **Start Robot World.command**, or run:

```sh
.venv/bin/python scripts/patch_viser.py
.venv/bin/python -m robot_world.web_app --open-browser
```

The viewer runs at **http://127.0.0.1:8765**. It stays local to this Mac and needs the Python process to remain running. No World Labs API key is sent to the browser. Local playback and commands do not spend API credits.

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
