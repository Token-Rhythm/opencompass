#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export _OPENCOMPASS_LANGUAGE_BENCHMARK=hmmt_2025

exec bash "$SCRIPT_DIR/run_language_benchmarks_smoke.sh" \
  --full \
  --model Qwen3.5-2B \
  --tokenizer-path Qwen/Qwen3.5-2B \
  --work-dir outputs/hmmt_2025_full_chat_qwen3.5_2b_thinking \
  --max-seq-len 65536 \
  --temperature 1.0 \
  --batch-size 512 \
  --max-workers 64 \
  --query-per-second 64 \
  --retry 1 \
  --timeout 3600 \
  --extra-body-json \
    '{"top_k":20,"min_p":0.0,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0}' \
  --openai-extra-kwargs-json '{}' \
  "$@"
