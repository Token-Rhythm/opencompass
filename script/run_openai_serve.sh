#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://47.116.36.105:8080/v1"
MODEL="Qwen3.5-9B"
API_KEY="EMPTY"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="outputs/vllm_api"
MAX_SEQ_LEN="65536"
MAX_OUT_LEN="65536"
TEMPERATURE="1.0"
QUERY_PER_SECOND="4"
BATCH_SIZE="16"
RETRY="1"
TIMEOUT="3600"
MAX_WORKERS="16"
EXTRA_BODY_JSON='{"top_k": 20, "repetition_penalty": 1.0}'
OPENAI_EXTRA_KWARGS_JSON='{"top_p": 0.95, "presence_penalty": 1.5}'
DEBUG_RUN="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_openai_serve.sh \
    --base-url http://127.0.0.1:8000/v1 \
    --model your-served-model-name \
    [--api-key EMPTY] \
    [--work-dir outputs/vllm_api_mmlu] \
    [--extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'] \
    [--openai-extra-kwargs-json '{"top_p":0.95,"seed":42}']

Options:
  --base-url                 vLLM OpenAI-compatible base URL, usually http://host:port/v1
  --model                    Model name exposed by vLLM
  --api-key                  API key, use EMPTY if your vLLM service does not check it
  --python                   Python executable, default: python or $PYTHON_BIN
  --work-dir                 OpenCompass output directory
  --max-seq-len              Max context length passed to OpenCompass model config
  --max-out-len              Max generation tokens
  --temperature              Sampling temperature
  --query-per-second         OpenCompass API rate limit
  --batch-size               OpenCompass batch size
  --retry                    API retry count
  --timeout                  Request timeout seconds
  --max-workers              Concurrent API worker threads
  --extra-body-json          JSON object passed as OpenAI SDK extra_body
  --openai-extra-kwargs-json JSON object merged into top-level chat.completions request
  --debug                    Run OpenCompass in debug mode
  --dry-run                  Print OpenCompass tasks without running them
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-out-len) MAX_OUT_LEN="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --query-per-second) QUERY_PER_SECOND="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --retry) RETRY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --extra-body-json) EXTRA_BODY_JSON="$2"; shift 2 ;;
    --openai-extra-kwargs-json) OPENAI_EXTRA_KWARGS_JSON="$2"; shift 2 ;;
    --debug) DEBUG_RUN="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$MODEL" ]]; then
  echo "Both --base-url and --model are required." >&2
  usage
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate your OpenCompass environment, or pass --python /path/to/python." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NORMALIZED_BASE_URL="${BASE_URL%/}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/chat/completions}"

EXTRA_BODY_PY="$($PYTHON_BIN - "$EXTRA_BODY_JSON" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
if data is not None and not isinstance(data, dict):
    raise SystemExit('--extra-body-json must be a JSON object')
print(repr(data or None))
PY
)"
OPENAI_EXTRA_KWARGS_PY="$($PYTHON_BIN - "$OPENAI_EXTRA_KWARGS_JSON" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
if data is not None and not isinstance(data, dict):
    raise SystemExit('--openai-extra-kwargs-json must be a JSON object')
print(repr(data or None))
PY
)"
BASE_URL_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$NORMALIZED_BASE_URL")"
API_KEY_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$API_KEY")"
MODEL_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$MODEL")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opencompass-api.XXXXXX")"
CONFIG_PATH="$TMP_DIR/vllm_api.py"

OC_VLLM_ABBR="$(printf '%s' "$MODEL" | sed 's/[^A-Za-z0-9_.-]/_/g')"
ABBR_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$OC_VLLM_ABBR")"

cat > "$CONFIG_PATH" <<PY
from mmengine.config import read_base
from opencompass.models import OpenAISDK

# with read_base():
#     from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        type=OpenAISDK,
        abbr=${ABBR_PY},
        path=${MODEL_PY},
        key=${API_KEY_PY},
        openai_api_base=${BASE_URL_PY},
        meta_template=api_meta_template,
        max_seq_len=${MAX_SEQ_LEN},
        max_out_len=${MAX_OUT_LEN},
        temperature=${TEMPERATURE},
        query_per_second=${QUERY_PER_SECOND},
        batch_size=${BATCH_SIZE},
        retry=${RETRY},
        timeout=${TIMEOUT},
        max_workers=${MAX_WORKERS},
        extra_body=${EXTRA_BODY_PY},
        openai_extra_kwargs=${OPENAI_EXTRA_KWARGS_PY},
    ),
]

datasets = aime2024_datasets
PY

cmd=(
  "$PYTHON_BIN" run.py
  "$CONFIG_PATH"
  --work-dir "$WORK_DIR"
  --max-num-workers 1
)

if [[ "$DEBUG_RUN" == "1" ]]; then
  cmd+=(--debug)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "OpenCompass config: $CONFIG_PATH"
echo "OpenAI-compatible base_url: $NORMALIZED_BASE_URL"
echo "Model: $MODEL"
exec "${cmd[@]}"
