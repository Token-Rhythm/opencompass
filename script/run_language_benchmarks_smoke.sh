#!/usr/bin/env bash
set -euo pipefail

# Smoke-test all phase-one and phase-two adapters through one OpenAI-compatible
# endpoint. INCLUDE uses vLLM prompt logprobs; the same endpoint is used as a
# temporary judge only for benchmarks that require one.

BASE_URL="http://47.116.36.105:8080/v1"
MODEL="Qwen3.5-9B"
API_KEY="EMPTY"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="outputs/language_benchmarks_smoke"

# AA-LCR inputs average about 100k tokens. The served model must expose at
# least this context length even though only a few samples are selected.
MAX_SEQ_LEN="131072"
AA_LCR_MIN_SEQ_LEN="131072"
MAX_OUT_LEN="2048"
JUDGE_MAX_OUT_LEN="256"
SAMPLES_PER_DATASET="2"
SUBSETS_PER_BENCHMARK="1"

TEMPERATURE="0.0"
QUERY_PER_SECOND="4"
BATCH_SIZE="4"
RETRY="1"
TIMEOUT="3600"
MAX_WORKERS="4"
EXTRA_BODY_JSON='{"top_k": 20, "repetition_penalty": 1.0}'
OPENAI_EXTRA_KWARGS_JSON='{"top_p": 0.95}'
DEBUG_RUN="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_language_benchmarks_smoke.sh \
    [--base-url http://127.0.0.1:8000/v1] \
    [--model your-served-model-name] \
    [--api-key EMPTY] \
    [--python /path/to/python] \
    [--work-dir outputs/language_benchmarks_smoke] \
    [--samples-per-dataset 2] \
    [--subsets-per-benchmark 1] \
    [--max-out-len 2048] \
    [--judge-max-out-len 256]

Benchmarks exercised:
  - MMLU-Redux 2.0
  - MMLU-ProX: first N language/category subsets
  - Global PIQA
  - INCLUDE base-44: first N language subsets
  - HMMT February 2025 and November 2025
  - PolyMath: first N language/difficulty subsets
  - MultiChallenge: first N samples, including LLM-judge and axis aggregation
  - AA-LCR: first N samples, including LLM equality judge

Important:
  This is a flow smoke test. INCLUDE follows its official raw continuation
  loglikelihood protocol and requires a vLLM endpoint with prompt_logprobs and
  return_token_ids support. MultiChallenge and AA-LCR temporarily use the
  tested model endpoint as judge, so their scores are not official. AA-LCR
  requires a 131072-token context. If /v1/models advertises a smaller
  max_model_len, this script reports the limitation and skips AA-LCR.
  The 2048/256-token defaults are still smoke-test limits. They are more
  likely to include a final answer than 256/128, but can still truncate hard
  reasoning tasks and must not be treated as official benchmark settings.
  For a small but more meaningful sanity run, pass --max-out-len 8192. For an
  official run, use each dataset config's own output limit instead of this
  script's shared override.

Options:
  --base-url                 OpenAI-compatible base URL
  --model                    Model name exposed by the endpoint
  --api-key                  API key, or EMPTY when authentication is disabled
  --python                   Python executable, default: python or $PYTHON_BIN
  --work-dir                 OpenCompass output directory
  --max-seq-len              Model context length; keep >=131072 for AA-LCR
  --max-out-len              Tested-model output tokens, default: 2048
  --judge-max-out-len        Temporary judge output tokens, default: 256
  --samples-per-dataset      Samples selected from each subset, default: 2
  --subsets-per-benchmark    Number of subsets for MMLU-ProX, INCLUDE and
                             PolyMath, or all; default: 1
  --temperature              Tested-model sampling temperature
  --query-per-second         OpenCompass API rate limit
  --batch-size               OpenCompass API batch size
  --retry                    API retry count
  --timeout                  Request timeout seconds
  --max-workers              Concurrent API worker threads
  --extra-body-json          JSON object passed as OpenAI SDK extra_body
  --openai-extra-kwargs-json JSON merged into chat.completions requests
  --debug                    Run OpenCompass in debug mode
  --dry-run                  Print tasks without running inference
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
    --judge-max-out-len) JUDGE_MAX_OUT_LEN="$2"; shift 2 ;;
    --samples-per-dataset) SAMPLES_PER_DATASET="$2"; shift 2 ;;
    --subsets-per-benchmark) SUBSETS_PER_BENCHMARK="$2"; shift 2 ;;
    --polymath-subsets) SUBSETS_PER_BENCHMARK="$2"; shift 2 ;;
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
  exit 1
fi

for numeric_value in MAX_SEQ_LEN MAX_OUT_LEN JUDGE_MAX_OUT_LEN SAMPLES_PER_DATASET; do
  value="${!numeric_value}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric_value must be a positive integer, got: $value" >&2
    exit 1
  fi
done

if [[ "$SUBSETS_PER_BENCHMARK" != "all" && ! "$SUBSETS_PER_BENCHMARK" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSETS_PER_BENCHMARK must be a positive integer or 'all'." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate OpenCompass, or pass --python /path/to/python." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NORMALIZED_BASE_URL="${BASE_URL%/}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/chat/completions}"

SERVED_MAX_SEQ_LEN="$($PYTHON_BIN - "$NORMALIZED_BASE_URL" "$API_KEY" "$MODEL" <<'PY'
import json
import sys
import urllib.request

base_url, api_key, model = sys.argv[1:]
request = urllib.request.Request(
    base_url.rstrip('/') + '/models',
    headers={'Authorization': f'Bearer {api_key}'},
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    match = next((item for item in payload.get('data', [])
                  if item.get('id') == model), None)
    if match is None:
        raise RuntimeError(f'model {model!r} is not listed by /v1/models')
    value = match.get('max_model_len')
    if not isinstance(value, int):
        raise RuntimeError('/v1/models did not return max_model_len')
    print(value)
except Exception as error:
    print(f'ERROR:{type(error).__name__}: {error}')
PY
)"

if [[ "$SERVED_MAX_SEQ_LEN" == ERROR:* ]]; then
  echo "vLLM endpoint preflight failed at ${NORMALIZED_BASE_URL}:" >&2
  echo "${SERVED_MAX_SEQ_LEN#ERROR:}" >&2
  echo "Start/restart the vLLM service, then rerun this command." >&2
  exit 1
fi

AA_LCR_SELECTION_PY="*aa_lcr_datasets,"
AA_LCR_SKIP_REASON=""
if (( MAX_SEQ_LEN < AA_LCR_MIN_SEQ_LEN )); then
  AA_LCR_SELECTION_PY=""
  AA_LCR_SKIP_REASON="configured --max-seq-len is ${MAX_SEQ_LEN}"
elif [[ "$SERVED_MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]] && (( SERVED_MAX_SEQ_LEN < AA_LCR_MIN_SEQ_LEN )); then
  AA_LCR_SELECTION_PY=""
  AA_LCR_SKIP_REASON="server advertises max_model_len=${SERVED_MAX_SEQ_LEN}"
fi

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
TEST_RANGE_PY="$($PYTHON_BIN -c 'import sys; print(repr("[:" + sys.argv[1] + "]"))' "$SAMPLES_PER_DATASET")"

if [[ "$SUBSETS_PER_BENCHMARK" == "all" ]]; then
  MMLU_PROX_SELECTION_PY="mmlu_prox_datasets"
  INCLUDE_SELECTION_PY="include_datasets"
  POLYMATH_SELECTION_PY="polymath_datasets"
else
  MMLU_PROX_SELECTION_PY="mmlu_prox_datasets[:${SUBSETS_PER_BENCHMARK}]"
  INCLUDE_SELECTION_PY="include_datasets[:${SUBSETS_PER_BENCHMARK}]"
  POLYMATH_SELECTION_PY="polymath_datasets[:${SUBSETS_PER_BENCHMARK}]"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opencompass-smoke.XXXXXX")"
CONFIG_PATH="$TMP_DIR/language_benchmarks_smoke.py"

OC_MODEL_ABBR="$(printf '%s' "$MODEL" | sed 's/[^A-Za-z0-9_.-]/_/g')"
ABBR_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$OC_MODEL_ABBR")"

cat > "$CONFIG_PATH" <<PY
from mmengine.config import read_base
from opencompass.models import OpenAISDK, VLLMOpenAIAPI
from opencompass.openicl.icl_inferencer import GenInferencer

with read_base():
    from opencompass.configs.datasets.aa_lcr.aa_lcr_gen import aa_lcr_datasets
    from opencompass.configs.datasets.global_piqa.global_piqa_generation import global_piqa_datasets
    from opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen import hmmt_2025_datasets
    from opencompass.configs.datasets.include.include_base_44_0shot_ppl import include_datasets
    from opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen import mmlu_prox_datasets
    from opencompass.configs.datasets.mmlu_redux.mmlu_redux_gen import mmlu_redux_datasets
    from opencompass.configs.datasets.multichallenge.multichallenge_gen import multichallenge_datasets
    from opencompass.configs.datasets.polymath.polymath_0shot_gen import polymath_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        type=VLLMOpenAIAPI,
        abbr=${ABBR_PY},
        path=${MODEL_PY},
        # This tokenizer is used only for conservative client-side length
        # estimates. INCLUDE scoring uses vLLM's returned prompt_token_ids.
        tokenizer_path='gpt-4',
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

# For smoke testing only: use the same endpoint as the temporary LLM judge.
# Official runs must restore the benchmark-specific judge configs.
smoke_judge_cfg = dict(
    type=OpenAISDK,
    abbr=${ABBR_PY} + '-smoke-judge',
    path=${MODEL_PY},
    tokenizer_path='gpt-4',
    key=${API_KEY_PY},
    openai_api_base=${BASE_URL_PY},
    meta_template=api_meta_template,
    max_seq_len=${MAX_SEQ_LEN},
    max_out_len=${JUDGE_MAX_OUT_LEN},
    temperature=0,
    query_per_second=${QUERY_PER_SECOND},
    batch_size=${BATCH_SIZE},
    retry=${RETRY},
    timeout=${TIMEOUT},
    max_workers=${MAX_WORKERS},
)

datasets = [
    *mmlu_redux_datasets,
    *${MMLU_PROX_SELECTION_PY},
    *global_piqa_datasets,
    *${INCLUDE_SELECTION_PY},
    *hmmt_2025_datasets,
    *${POLYMATH_SELECTION_PY},
    *multichallenge_datasets,
    ${AA_LCR_SELECTION_PY}
]

for dataset in datasets:
    dataset['abbr'] = f"{dataset['abbr']}_smoke"
    dataset['reader_cfg']['test_range'] = ${TEST_RANGE_PY}
    if 'max_out_len' in dataset['infer_cfg']['inferencer']:
        dataset['infer_cfg']['inferencer']['max_out_len'] = ${MAX_OUT_LEN}

    evaluator = dataset['eval_cfg']['evaluator']
    evaluator_dataset = evaluator.get('dataset_cfg')
    if evaluator_dataset is not None:
        evaluator_dataset['reader_cfg']['test_range'] = ${TEST_RANGE_PY}
        evaluator['judge_cfg'] = smoke_judge_cfg
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
echo "Benchmarks: MMLU-Redux, MMLU-ProX, Global PIQA, INCLUDE, HMMT Feb/Nov 2025, PolyMath, MultiChallenge, AA-LCR"
echo "Subsets per multilingual benchmark: $SUBSETS_PER_BENCHMARK"
echo "Samples per subset: $SAMPLES_PER_DATASET"
echo "Model/Judge max output tokens: $MAX_OUT_LEN/$JUDGE_MAX_OUT_LEN"
if (( MAX_OUT_LEN <= 2048 )); then
  echo "WARNING: outputs at this limit can end before the final answer; use 8192 for a stronger sanity run."
fi
if [[ -n "$AA_LCR_SKIP_REASON" ]]; then
  echo "SKIP: AA-LCR requires ${AA_LCR_MIN_SEQ_LEN} tokens; ${AA_LCR_SKIP_REASON}."
fi
echo "WARNING: smoke-test judges are not the official benchmark judges."
exec "${cmd[@]}"
