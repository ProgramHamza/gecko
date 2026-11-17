#!/usr/bin/env bash
# Usage: ./run.sh <command> [options]   e.g. ./run.sh run --days 10
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "No .venv found. Run: bash setup.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec python -m devlm "$@"
