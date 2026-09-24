#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$REPO_ROOT/tools/run_mmmu_pro.py" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/../opencompass-vision" && pwd)"
fi
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/../opencompass-venv/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$REPO_ROOT/tools/run_mmmu_pro.py" "$@"
