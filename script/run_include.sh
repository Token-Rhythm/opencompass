#!/usr/bin/env bash
set -euo pipefail

# INCLUDE is a continuation-loglikelihood benchmark, not a chat-generation
# benchmark.  Select only its 44 language tasks and let the shared launcher
# route them to /v1/completions with prompt_logprobs/return_token_ids.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export _OPENCOMPASS_LANGUAGE_BENCHMARK=include

exec bash "$SCRIPT_DIR/run_language_benchmarks_smoke.sh" \
  --full \
  --model Qwen3.5-2B \
  --tokenizer-path Qwen/Qwen3.5-2B \
  --work-dir outputs/include_base_44_qwen3.5_2b \
  --max-seq-len 65536 \
  --subsets-per-benchmark all \
  --samples-per-dataset all \
  --batch-size 512 \
  --max-workers 64 \
  --query-per-second 64 \
  --retry 1 \
  --timeout 3600 \
  --extra-body-json '{}' \
  --openai-extra-kwargs-json '{}' \
  "$@"
