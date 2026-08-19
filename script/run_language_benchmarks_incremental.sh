#!/usr/bin/env bash
set -euo pipefail

# Run the language benchmark suite one benchmark at a time. The underlying
# launcher writes a score after every benchmark and a combined summary after
# the final benchmark.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_language_benchmarks_smoke.sh" \
  --incremental-eval "$@"
