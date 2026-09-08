#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Set up the Python environment first; see README.md.'
  exit 1
fi
command -v cargo >/dev/null || { print 'Install Rust first; see realsense/README.md.'; exit 1; }

# Reuse only this checkout's Robot World server, never an unrelated port owner.
state=$(.venv/bin/python - <<'PY'
import pathlib
import socket
import subprocess

root = pathlib.Path.cwd()
with socket.socket() as probe:
    probe.settimeout(1)
    if probe.connect_ex(('127.0.0.1', 8765)) != 0:
        print('start')
        raise SystemExit
result = subprocess.run(['/usr/sbin/lsof', '-nP', '-iTCP:8765', '-sTCP:LISTEN', '-t'], capture_output=True, text=True)
for pid in set(result.stdout.split()):
    command = subprocess.run(['/bin/ps', '-p', pid, '-o', 'command='], capture_output=True, text=True).stdout
    cwd = subprocess.run(['/usr/sbin/lsof', '-a', '-p', pid, '-d', 'cwd', '-Fn'], capture_output=True, text=True).stdout
    if '-m robot_world.web_app' in command and f'n{root}' in cwd.splitlines():
        print('reuse')
        raise SystemExit
print('occupied')
PY
)
case "$state" in
  start)
    open -a Terminal "$PWD/Start Robot World.command"
    ;;
  reuse)
    print 'Robot World is already running; opening its viewer.'
    open 'http://127.0.0.1:8765'
    ;;
  *)
    print 'Port 8765 is occupied by another process. Close it before launching Robot World.'
    exit 1
    ;;
esac

# Keep the camera's sudo prompt in this Terminal window. Its existing launcher
# handles rebuilds, replacement of old helpers, and cleanup when the viewer exits.
exec ./realsense/scripts/start.sh "$@"
