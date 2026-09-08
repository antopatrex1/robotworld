#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
exec ./scripts/start.sh "$@"
