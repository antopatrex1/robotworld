#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'Set up the Python environment first; see README.md.'
  exit 1
fi
.venv/bin/python scripts/patch_viser.py || exit 1
exec .venv/bin/python -m robot_world.web_app --open-browser "$@"
