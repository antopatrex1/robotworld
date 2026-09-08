# RealSense Studio

Robot World's native Rust / GPUI RGB-D camera window and stdio MCP server, imported from `~/Projects/codex/oai6`. Robot World's browser interface also displays live color and depth from this component in its **RealSense · live camera** panel. The native window, browser preview, and MCP server share one camera helper. Registering the camera to the simulated room and using its observations for robot actions remain later steps.

## Launch

From the Robot World repository root:

```sh
./realsense/scripts/start.sh
```

Or open **Start RealSense.command** in this folder. Enter your Mac password at the Terminal prompt: the camera helper needs administrator access for macOS USB access. The window and MCP server run as your normal user. The D435 streams 640×480 RGB8 and Z16 at 30 fps. Closing the window stops the helper started by the launcher.

Dependencies: Rust (edition 2024), Xcode command-line tools / Metal compiler, librealsense, and pkg-config. Install the native libraries with `brew install librealsense pkg-config`. The first build can take several minutes. The included build script locates Homebrew's actual macOS library directory.

For a synthetic preview without a camera or sudo:

```sh
./realsense/scripts/start.sh --demo
```

Demo frames are explicitly labeled. Close the original `oai6` camera helper before launching the hardware helper here: both checkouts would otherwise compete for the same USB camera. This launch script only replaces helpers belonging to this checkout.

## Capture

Click **Capture RGB + depth**, call the MCP tool, or run:

```sh
./realsense/target/debug/realsense-studio capture
```

Each capture waits for a fresh frameset and saves a unique directory under `realsense/captures/` containing:

- `color.png`: RGB image.
- `depth.png`: lossless 16-bit grayscale depth; multiply each pixel by `depth_scale_m` to obtain metres. Zero means invalid/no return.
- `depth-preview.png`: depth visualization on a fixed 0.2–4 m scale; invalid pixels are black.
- `metadata.json`: device serial, dimensions, frame numbers, device timestamps, host receipt time, depth scale, demo flag, and file paths.

Depth uses native camera coordinates and is **not pixel-aligned to color**. Both streams come from one frameset, but their exposures are not necessarily simultaneous. Raw depth has no filtering or hole filling.

## MCP server

Build once with `cargo build --locked --manifest-path realsense/Cargo.toml`, then configure your MCP client using [mcp.example.json](mcp.example.json). Replace the absolute command path if you move the checkout. Start the camera helper using the launcher before calling capture tools; the MCP process connects to that shared helper.

The server exposes `camera_status` and `capture_rgbd`. Capture returns inline color and colorized depth PNGs, plus metadata and paths to the raw depth files. MCP stdout contains protocol messages only.

To run a hardware helper without a window, from the repository root:

```sh
mkdir -p realsense/.runtime
chmod 700 realsense/.runtime
sudo ./realsense/target/debug/realsense-studio camera
```

Use `camera --demo` without sudo for synthetic headless capture. Ctrl-C stops the foreground helper. Run `./realsense/target/debug/realsense-studio mcp` for stdio service, or `status` to inspect the helper.

Default socket and capture paths are anchored to this folder at compile time, independently of the caller's working directory. Rebuild after moving the checkout. `--socket PATH` and `--output PATH` override defaults; all clients must use the same socket. The socket is mode 0600 inside a mode 0700 runtime directory. Only unprivileged clients write captures. Logs are in `realsense/.runtime/camera.log` when using the launcher.

## Verification and app bundle

```sh
cd realsense
cargo test --locked
python3 scripts/test_mcp.py
cargo clippy --locked --all-targets -- -D warnings
./scripts/bundle.sh
```

The Rust tests cover RGB channel conversion and lossless depth capture with unique output directories. The MCP integration test uses an isolated synthetic helper to verify parent-process cleanup, tool discovery, fresh capture, inline PNGs, raw Z16 depth, metadata, and offline errors. Hardware streaming requires a connected D435 and an elevated helper.

The bundle appears at `realsense/dist/RealSense Studio.app` and requires a running helper. It depends on this Mac's Homebrew libraries. Build output, runtime sockets/logs, captures, and generated app bundles are ignored by Git.
