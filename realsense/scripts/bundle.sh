#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
cargo build --locked
bundle="$PWD/dist/RealSense Studio.app"
mkdir -p "$bundle/Contents/MacOS"
cp target/debug/realsense-studio "$bundle/Contents/MacOS/realsense-studio"
cp assets/Info.plist "$bundle/Contents/Info.plist"
codesign --force --sign - "$bundle"
echo "$bundle"
