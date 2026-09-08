#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
mkdir -p .runtime captures
chmod 700 .runtime
cargo build --locked
binary="$PWD/target/debug/realsense-studio"
# Replace only this checkout's default-socket helper, not sudo's wrappers.
old_helpers=()
while read -r candidate_pid candidate_command; do
  if [[ "$candidate_command" == "$binary camera" ||
        "$candidate_command" == "$binary camera --demo" ||
        "$candidate_command" == "$binary camera --parent-pid "<-> ||
        "$candidate_command" == "$binary camera --demo --parent-pid "<-> ]]; then
    old_helpers+=("$candidate_pid")
  fi
done < <(ps -axo pid=,command=)
if [[ "${1:-}" != "--demo" || ${#old_helpers} -gt 0 ]]; then
  echo "Authorizing the RealSense USB helper."
  sudo -v
fi
for old_pid in "${old_helpers[@]}"; do
  echo "Stopping previous camera helper (PID $old_pid)."
  sudo -n kill -TERM "$old_pid"
  for attempt in {1..50}; do
    if ! ps -p "$old_pid" >/dev/null 2>&1; then break; fi
    sleep 0.1
  done
  if ps -p "$old_pid" >/dev/null 2>&1; then
    echo "Previous helper did not exit; refusing to open the camera twice." >&2
    exit 1
  fi
done
if [[ "${1:-}" == "--demo" ]]; then
  "$binary" camera --demo --parent-pid $$ >.runtime/camera.log 2>&1 &
else
  sudo -n "$binary" camera --parent-pid $$ >.runtime/camera.log 2>&1 &
fi
# The helper exits when this launcher ends, even if sudo authentication expires.
trap 'exit 130' INT
trap 'exit 143' TERM
"$binary" viewer
