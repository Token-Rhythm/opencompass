#!/usr/bin/env bash
set -euo pipefail

# Run all phase-one and phase-two adapters through one OpenAI-compatible
# endpoint. The default is a small smoke test; --full selects every subset and
# every sample without overriding the benchmark-specific output limits.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_URL="http://47.116.36.105:8080/v1"
MODEL="Qwen3.5-9B"
API_KEY="EMPTY"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/../opencompass-venv/bin/python}"
TOKENIZER_PATH="Qwen/Qwen3.5-2B"
WORK_DIR="outputs/language_benchmarks_smoke"

# AA-LCR inputs average about 100k tokens. The served model must expose at
# least this context length even though only a few samples are selected.
MAX_SEQ_LEN="131072"
AA_LCR_MIN_SEQ_LEN="131072"
MAX_OUT_LEN="2048"
MAX_OUT_LEN_EXPLICIT="0"
JUDGE_MAX_OUT_LEN="256"
SAMPLES_PER_DATASET="2"
SUBSETS_PER_BENCHMARK="1"
GENERATION_ENDPOINT="chat"

TEMPERATURE="1.0"
QUERY_PER_SECOND="4"
BATCH_SIZE="4"
RETRY="1"
TIMEOUT="3600"
MAX_WORKERS="4"
EXTRA_BODY_JSON='{"top_k": 20, "repetition_penalty": 1.0}'
OPENAI_EXTRA_KWARGS_JSON='{"top_p": 0.95}'
STREAM_RESPONSES="1"
DEBUG_RUN="0"
DRY_RUN="0"
FULL_RUN="0"
REUSE_RUN=""
INCREMENTAL_EVAL="0"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_language_benchmarks_smoke.sh \
    [--base-url http://127.0.0.1:8000/v1] \
    [--model your-served-model-name] \
    [--api-key EMPTY] \
    [--python /path/to/python] \
    [--tokenizer-path Qwen/Qwen3.5-2B] \
    [--work-dir outputs/language_benchmarks_smoke] \
    [--samples-per-dataset 2] \
    [--subsets-per-benchmark 1] \
    [--generation-endpoint chat] \
    [--reuse 20260724_013120] \
    [--incremental-eval] \
    [--max-out-len 2048] \
    [--judge-max-out-len 256]

Full-data example for a post-trained model:
  bash script/run_language_benchmarks_smoke.sh --full \
    --generation-endpoint chat \
    --work-dir outputs/language_benchmarks_full_chat

Incremental-evaluation example (score each benchmark immediately):
  bash script/run_language_benchmarks_incremental.sh --full \
    --generation-endpoint chat \
    --work-dir outputs/language_benchmarks_incremental_chat

Benchmarks exercised:
  - MMLU-Redux 2.0
  - MMLU-ProX Full/Lite, zero-shot/5-shot: first N subsets per variant
  - Global PIQA
  - INCLUDE base-44: first N language subsets
  - HMMT February 2025 and November 2025
  - PolyMath: first N language/difficulty subsets
  - MultiChallenge: first N samples, including LLM-judge and axis aggregation
  - AA-LCR: first N samples, including LLM equality judge

Important:
  All generation/chat inferencers use /v1/chat/completions so post-trained
  models receive their chat template. Only probability-based inferencers
  (LL/PPL and related types) use /v1/completions. In this suite that means
  INCLUDE retains its dedicated prompt-logprob path, while MMLU-Redux,
  MMLU-ProX, Global PIQA, HMMT, PolyMath, MultiChallenge and AA-LCR use chat.
  Without --full this is a flow smoke test. INCLUDE follows its official raw
  continuation loglikelihood protocol and requires a vLLM endpoint with
  prompt_logprobs and return_token_ids support. MultiChallenge and AA-LCR
  temporarily use the tested model endpoint as judge in both modes, so their
  scores are not official. AA-LCR
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
  --python                   Python executable; defaults to the sibling
                             ../opencompass-venv environment
  --tokenizer-path           Tokenizer used for context accounting
  --work-dir                 OpenCompass output directory
  --max-seq-len              Model context length; keep >=131072 for AA-LCR
  --max-out-len              Tested-model output tokens, default: 2048
  --judge-max-out-len        Temporary judge output tokens, default: 256
  --samples-per-dataset      Samples selected from each subset, default: 2
  --subsets-per-benchmark    Number of subsets for MMLU-ProX, INCLUDE and
                             PolyMath, or all; default: 1
  --generation-endpoint      Compatibility option; must be chat. Generation
                             tasks no longer support raw completions.
  --temperature              Tested-model sampling temperature
  --query-per-second         OpenCompass API rate limit
  --batch-size               OpenCompass API batch size
  --retry                    API retry count
  --timeout                  Request timeout seconds
  --max-workers              Concurrent API worker threads
  --extra-body-json          JSON object passed as OpenAI SDK extra_body
  --openai-extra-kwargs-json JSON merged into generation requests
  --stream-responses        Stream chat tokens (default; avoids buffering
                            long thinking responses)
  --no-stream-responses     Disable streaming for compatibility debugging
  --debug                    Run OpenCompass in debug mode
  --dry-run                  Print tasks without running inference
  --reuse                    Reuse a prior timestamp below --work-dir and run
                             only its missing jobs
  --incremental-eval         Run infer -> eval -> summary for one benchmark at
                             a time, then generate a combined final summary
  --full                     Select all subsets/samples and retain each
                             benchmark's configured output-token limit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-out-len)
      MAX_OUT_LEN="$2"
      MAX_OUT_LEN_EXPLICIT="1"
      shift 2
      ;;
    --judge-max-out-len) JUDGE_MAX_OUT_LEN="$2"; shift 2 ;;
    --samples-per-dataset) SAMPLES_PER_DATASET="$2"; shift 2 ;;
    --subsets-per-benchmark) SUBSETS_PER_BENCHMARK="$2"; shift 2 ;;
    --generation-endpoint) GENERATION_ENDPOINT="$2"; shift 2 ;;
    # Backward-compatible alias used by earlier versions of this script.
    --mmlu-prox-endpoint) GENERATION_ENDPOINT="$2"; shift 2 ;;
    --polymath-subsets) SUBSETS_PER_BENCHMARK="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --query-per-second) QUERY_PER_SECOND="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --retry) RETRY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --extra-body-json) EXTRA_BODY_JSON="$2"; shift 2 ;;
    --openai-extra-kwargs-json) OPENAI_EXTRA_KWARGS_JSON="$2"; shift 2 ;;
    --stream-responses) STREAM_RESPONSES="1"; shift ;;
    --no-stream-responses) STREAM_RESPONSES="0"; shift ;;
    --debug) DEBUG_RUN="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --reuse) REUSE_RUN="$2"; shift 2 ;;
    --incremental-eval) INCREMENTAL_EVAL="1"; shift ;;
    --full) FULL_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$FULL_RUN" == "1" ]]; then
  SAMPLES_PER_DATASET="all"
  SUBSETS_PER_BENCHMARK="all"
fi

if [[ "$MAX_OUT_LEN_EXPLICIT" == "1" ]]; then
  OVERRIDE_MAX_OUT_LEN_PY="True"
else
  OVERRIDE_MAX_OUT_LEN_PY="False"
fi

if [[ "$STREAM_RESPONSES" == "1" ]]; then
  STREAM_RESPONSES_PY="True"
else
  STREAM_RESPONSES_PY="False"
fi

if [[ -z "$BASE_URL" || -z "$MODEL" ]]; then
  echo "Both --base-url and --model are required." >&2
  exit 1
fi

for numeric_value in MAX_SEQ_LEN MAX_OUT_LEN JUDGE_MAX_OUT_LEN; do
  value="${!numeric_value}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric_value must be a positive integer, got: $value" >&2
    exit 1
  fi
done

if [[ "$SAMPLES_PER_DATASET" != "all" && ! "$SAMPLES_PER_DATASET" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLES_PER_DATASET must be a positive integer or 'all'." >&2
  exit 1
fi

if [[ "$SUBSETS_PER_BENCHMARK" != "all" && ! "$SUBSETS_PER_BENCHMARK" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSETS_PER_BENCHMARK must be a positive integer or 'all'." >&2
  exit 1
fi

if [[ "$GENERATION_ENDPOINT" != "chat" ]]; then
  echo "--generation-endpoint must be 'chat' for post-trained benchmarks." >&2
  echo "Probability-based datasets are routed to completions automatically." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate OpenCompass, or pass --python /path/to/python." >&2
  exit 1
fi

cd "$REPO_ROOT"

NORMALIZED_BASE_URL="${BASE_URL%/}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/chat/completions}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/completions}"

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
TOKENIZER_PATH_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$TOKENIZER_PATH")"
if [[ "$SAMPLES_PER_DATASET" == "all" ]]; then
  TEST_RANGE_PY="None"
else
  TEST_RANGE_PY="$($PYTHON_BIN -c 'import sys; print(repr("[:" + sys.argv[1] + "]"))' "$SAMPLES_PER_DATASET")"
fi
if [[ "$FULL_RUN" == "1" ]]; then
  FULL_RUN_PY="True"
  RUN_SUFFIX="full_${GENERATION_ENDPOINT}"
else
  FULL_RUN_PY="False"
  RUN_SUFFIX="smoke_${GENERATION_ENDPOINT}"
fi
if [[ "$SUBSETS_PER_BENCHMARK" == "all" ]]; then
  MMLU_PROX_5SHOT_SELECTION_PY="mmlu_prox_5shot_datasets"
  MMLU_PROX_0SHOT_SELECTION_PY="mmlu_prox_0shot_datasets"
  MMLU_PROX_LITE_5SHOT_SELECTION_PY="mmlu_prox_lite_5shot_datasets"
  MMLU_PROX_LITE_0SHOT_SELECTION_PY="mmlu_prox_lite_0shot_datasets"
  INCLUDE_SELECTION_PY="include_datasets"
  POLYMATH_SELECTION_PY="polymath_datasets"
else
  MMLU_PROX_5SHOT_SELECTION_PY="mmlu_prox_5shot_datasets[:${SUBSETS_PER_BENCHMARK}]"
  MMLU_PROX_0SHOT_SELECTION_PY="mmlu_prox_0shot_datasets[:${SUBSETS_PER_BENCHMARK}]"
  MMLU_PROX_LITE_5SHOT_SELECTION_PY="mmlu_prox_lite_5shot_datasets[:${SUBSETS_PER_BENCHMARK}]"
  MMLU_PROX_LITE_0SHOT_SELECTION_PY="mmlu_prox_lite_0shot_datasets[:${SUBSETS_PER_BENCHMARK}]"
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
    from opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen import mmlu_prox_5shot_datasets
    from opencompass.configs.datasets.mmlu_prox.mmlu_prox_0shot_cot_gen import mmlu_prox_0shot_datasets
    from opencompass.configs.datasets.mmlu_prox.mmlu_prox_lite_5shot_cot_gen import mmlu_prox_lite_5shot_datasets
    from opencompass.configs.datasets.mmlu_prox.mmlu_prox_lite_0shot_cot_gen import mmlu_prox_lite_0shot_datasets
    from opencompass.configs.datasets.mmlu_redux.mmlu_redux_gen import mmlu_redux_datasets, mmlu_redux_lm_eval_datasets
    from opencompass.configs.datasets.multichallenge.multichallenge_gen import multichallenge_datasets
    from opencompass.configs.datasets.polymath.polymath_0shot_gen import polymath_datasets

test_range = ${TEST_RANGE_PY}

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

common_model_cfg = dict(
    type=VLLMOpenAIAPI,
    path=${MODEL_PY},
    # This tokenizer is used only for conservative client-side length
    # estimates. INCLUDE scoring uses vLLM's returned prompt_token_ids.
    tokenizer_path=${TOKENIZER_PATH_PY},
    key=${API_KEY_PY},
    openai_api_base=${BASE_URL_PY},
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
    stream_chat=${STREAM_RESPONSES_PY},
)

# Keep distinct execution abbreviations so LocalRunner never reuses a model
# object with the wrong endpoint. Raw completions is reserved for datasets
# that inspect logprobs; every generated answer uses the chat endpoint. The
# shared summarizer_abbr presents both routes as one model in the summary.
raw_model = dict(
    **common_model_cfg,
    abbr=${ABBR_PY} + '-completions',
    summarizer_abbr=${ABBR_PY},
    generation_endpoint='completions',
    meta_template=None,
)
chat_model = dict(
    **common_model_cfg,
    abbr=${ABBR_PY} + '-chat',
    summarizer_abbr=${ABBR_PY},
    generation_endpoint='chat',
    meta_template=api_meta_template,
)
models = [
    raw_model,
    chat_model,
]

# For smoke testing only: use the same endpoint as the temporary LLM judge.
# Official runs must restore the benchmark-specific judge configs.
smoke_judge_cfg = dict(
    type=OpenAISDK,
    abbr=${ABBR_PY} + '-smoke-judge',
    path=${MODEL_PY},
    tokenizer_path=${TOKENIZER_PATH_PY},
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
    *mmlu_redux_lm_eval_datasets,
    *${MMLU_PROX_5SHOT_SELECTION_PY},
    *${MMLU_PROX_0SHOT_SELECTION_PY},
    *${MMLU_PROX_LITE_5SHOT_SELECTION_PY},
    *${MMLU_PROX_LITE_0SHOT_SELECTION_PY},
    *global_piqa_datasets,
    *${INCLUDE_SELECTION_PY},
    *hmmt_2025_datasets,
    *${POLYMATH_SELECTION_PY},
    *multichallenge_datasets,
    ${AA_LCR_SELECTION_PY}
]

for dataset in datasets:
    dataset['abbr'] = f"{dataset['abbr']}_${RUN_SUFFIX}"
    if test_range is not None:
        dataset['reader_cfg']['test_range'] = test_range
    if ((not ${FULL_RUN_PY} or ${OVERRIDE_MAX_OUT_LEN_PY})
            and 'max_out_len' in dataset['infer_cfg']['inferencer']):
        dataset['infer_cfg']['inferencer']['max_out_len'] = ${MAX_OUT_LEN}

    evaluator = dataset['eval_cfg']['evaluator']
    evaluator_dataset = evaluator.get('dataset_cfg')
    if evaluator_dataset is not None:
        if test_range is not None:
            evaluator_dataset['reader_cfg']['test_range'] = test_range
        evaluator['judge_cfg'] = smoke_judge_cfg

# MMEngine represents imports as lazy objects while loading this generated
# config, so endpoint groups are declared explicitly. The routing tests verify
# that INCLUDE remains probability-based and every dataset in the second group
# uses a generation/chat inferencer.
logprob_datasets = [*${INCLUDE_SELECTION_PY}]
chat_datasets = [
    *mmlu_redux_datasets,
    *mmlu_redux_lm_eval_datasets,
    *${MMLU_PROX_5SHOT_SELECTION_PY},
    *${MMLU_PROX_0SHOT_SELECTION_PY},
    *${MMLU_PROX_LITE_5SHOT_SELECTION_PY},
    *${MMLU_PROX_LITE_0SHOT_SELECTION_PY},
    *global_piqa_datasets,
    *hmmt_2025_datasets,
    *${POLYMATH_SELECTION_PY},
    *multichallenge_datasets,
    ${AA_LCR_SELECTION_PY}
]
model_dataset_combinations = [
    dict(models=[raw_model], datasets=logprob_datasets),
    dict(models=[chat_model], datasets=chat_datasets),
]

# The incremental launcher selects exactly one benchmark per OpenCompass
# invocation. Keeping the selection in this generated config lets every stage
# use the same dataset/model objects and therefore the same prediction paths.
benchmark_dataset_groups = {
    'mmlu_redux': [*mmlu_redux_datasets],
    'mmlu_redux_lm_eval': [*mmlu_redux_lm_eval_datasets],
    'include': [*${INCLUDE_SELECTION_PY}],
    'mmlu_prox': [
        *${MMLU_PROX_5SHOT_SELECTION_PY},
        *${MMLU_PROX_0SHOT_SELECTION_PY},
        *${MMLU_PROX_LITE_5SHOT_SELECTION_PY},
        *${MMLU_PROX_LITE_0SHOT_SELECTION_PY},
    ],
    'global_piqa': [*global_piqa_datasets],
    'hmmt_2025': [*hmmt_2025_datasets],
    'hmmt_feb_2025': [hmmt_2025_datasets[0]],
    'hmmt_nov_2025': [hmmt_2025_datasets[1]],
    'polymath': [*${POLYMATH_SELECTION_PY}],
    'multichallenge': [*multichallenge_datasets],
    'aa_lcr': [${AA_LCR_SELECTION_PY}],
}
benchmark_model_groups = {
    'mmlu_redux': [chat_model],
    'mmlu_redux_lm_eval': [chat_model],
    'include': [raw_model],
    'mmlu_prox': [chat_model],
    'global_piqa': [chat_model],
    'hmmt_2025': [chat_model],
    'hmmt_feb_2025': [chat_model],
    'hmmt_nov_2025': [chat_model],
    'polymath': [chat_model],
    'multichallenge': [chat_model],
    'aa_lcr': [chat_model],
}

selected_benchmark = __import__('os').environ.get(
    '_OPENCOMPASS_LANGUAGE_BENCHMARK', '').strip()
if selected_benchmark:
    if selected_benchmark not in benchmark_dataset_groups:
        raise ValueError(
            f'Unknown incremental benchmark: {selected_benchmark!r}')
    datasets = benchmark_dataset_groups[selected_benchmark]
    model_dataset_combinations = [
        dict(
            models=benchmark_model_groups[selected_benchmark],
            datasets=datasets,
        )
    ]

# INCLUDE's headline number is the macro-average accuracy over its language
# tasks.  Keep each language visible while adding the aggregate row expected
# by benchmark tables.  The generated suffix is already part of each dataset
# abbreviation at this point, so the group also works for smoke/full runs.
include_summary_subsets = [
    dataset['abbr'] for dataset in datasets
    if dataset['abbr'].startswith('include_base_44_')
]
include_summary_groups = []
summary_dataset_abbrs = [dataset['abbr'] for dataset in datasets]
if include_summary_subsets:
    include_summary_groups.append(
        dict(name='include_base_44', subsets=include_summary_subsets))
    summary_dataset_abbrs = ['include_base_44', *summary_dataset_abbrs]
summarizer = dict(
    dataset_abbrs=summary_dataset_abbrs,
    summary_groups=include_summary_groups,
)
del benchmark_dataset_groups
del benchmark_model_groups
del selected_benchmark
del logprob_datasets
del chat_datasets
del include_summary_subsets
del include_summary_groups
del summary_dataset_abbrs
PY

ROUTING_SUMMARY="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import collections
import sys

from mmengine.config import Config


# Keep this launch policy local to the launcher: OpenCompass inferencers that
# score token probabilities require /v1/completions; all others use chat.
LOGPROB_INFERENCERS = {
    'CLPInferencer',
    'InferencePPLOnlyInferencer',
    'LLInferencer',
    'MinKPercentInferencer',
    'PPLInferencer',
    'PPLOnlyInferencer',
    'SWCELossInferencer',
}


def inferencer_uses_logprobs(dataset):
    inferencer_type = dataset['infer_cfg']['inferencer']['type']
    if isinstance(inferencer_type, str):
        type_name = inferencer_type.rsplit('.', 1)[-1]
    else:
        type_name = inferencer_type.__name__
    return type_name in LOGPROB_INFERENCERS

config = Config.fromfile(sys.argv[1], format_python_code=False)
routed_abbrs = []
route_counts = collections.Counter()
for combination in config.model_dataset_combinations:
    datasets = combination['datasets']
    requirements = {inferencer_uses_logprobs(dataset) for dataset in datasets}
    if len(requirements) != 1:
        raise SystemExit('A model combination mixes logprob and generation '
                         'datasets')
    expected_endpoint = 'completions' if requirements == {True} else 'chat'
    for model in combination['models']:
        actual_endpoint = model.get('generation_endpoint')
        if actual_endpoint != expected_endpoint:
            raise SystemExit(
                f"Invalid route for {model.get('abbr')}: expected "
                f'{expected_endpoint}, got {actual_endpoint}')
    routed_abbrs.extend(dataset['abbr'] for dataset in datasets)
    route_counts[expected_endpoint] += len(datasets)

expected_abbrs = [dataset['abbr'] for dataset in config.datasets]
if collections.Counter(routed_abbrs) != collections.Counter(expected_abbrs):
    raise SystemExit('Endpoint groups do not cover every selected dataset '
                     'exactly once')
print(f"chat={route_counts['chat']}, "
      f"completions(logprob)={route_counts['completions']}")
PY
)"

cmd=(
  "$PYTHON_BIN" run.py
  "$CONFIG_PATH"
  --work-dir "$WORK_DIR"
  # Keep one OpenCompass task process. API request concurrency is controlled
  # independently by the model's max_workers/batch_size; coupling the two
  # makes every task process reload the same Hugging Face dataset.
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
echo "Generation routing: all generated answers=chat; logprob datasets=completions"
echo "Stream chat responses: $STREAM_RESPONSES"
echo "Validated dataset routes: $ROUTING_SUMMARY"
echo "Benchmarks: MMLU-Redux, MMLU-ProX, Global PIQA, INCLUDE, HMMT Feb/Nov 2025, PolyMath, MultiChallenge, AA-LCR"
echo "Subsets per multilingual benchmark: $SUBSETS_PER_BENCHMARK"
echo "Samples per subset: $SAMPLES_PER_DATASET"
if [[ "$FULL_RUN" == "1" ]]; then
  if [[ "$MAX_OUT_LEN_EXPLICIT" == "1" ]]; then
    echo "Run mode: full data; generation max output tokens overridden to $MAX_OUT_LEN"
  else
    echo "Run mode: full data; benchmark-specific generation limits retained"
  fi
else
  echo "Run mode: smoke; model/judge max output tokens: $MAX_OUT_LEN/$JUDGE_MAX_OUT_LEN"
fi
if [[ "$FULL_RUN" != "1" ]] && (( MAX_OUT_LEN <= 2048 )); then
  echo "WARNING: outputs at this limit can end before the final answer; use 8192 for a stronger sanity run."
fi
if [[ -n "$AA_LCR_SKIP_REASON" ]]; then
  echo "SKIP: AA-LCR requires ${AA_LCR_MIN_SEQ_LEN} tokens; ${AA_LCR_SKIP_REASON}."
fi
echo "WARNING: smoke-test judges are not the official benchmark judges."

if [[ "$INCREMENTAL_EVAL" != "1" ]]; then
  if [[ -n "$REUSE_RUN" ]]; then
    cmd+=(--reuse "$REUSE_RUN")
  fi
  exec "${cmd[@]}"
fi

# Use an explicit run id from the first invocation onward. OpenCompass accepts
# a not-yet-created --reuse directory and creates it, which keeps all
# per-benchmark predictions/results under one experiment directory.
if [[ -n "$REUSE_RUN" ]]; then
  INCREMENTAL_RUN_ID="$REUSE_RUN"
else
  INCREMENTAL_RUN_ID="$(date '+%Y%m%d_%H%M%S')"
fi
INCREMENTAL_RUN_DIR="$WORK_DIR/$INCREMENTAL_RUN_ID"
INCREMENTAL_BENCHMARKS=(
  mmlu_redux
  include
  mmlu_prox
  global_piqa
  hmmt_2025
  polymath
  multichallenge
  aa_lcr
)

echo "Incremental evaluation: enabled"
echo "Incremental run id: $INCREMENTAL_RUN_ID"
echo "Incremental run directory: $INCREMENTAL_RUN_DIR"

copy_new_summaries() {
  local marker="$1"
  local destination="$2"
  local summary_root="$INCREMENTAL_RUN_DIR/summary"

  [[ -d "$summary_root" ]] || return 0
  mkdir -p "$destination"
  find "$summary_root" -maxdepth 1 -type f -newer "$marker" \
    -exec cp -p {} "$destination/" \;
}

for benchmark in "${INCREMENTAL_BENCHMARKS[@]}"; do
  if [[ "$benchmark" == "aa_lcr" && -n "$AA_LCR_SKIP_REASON" ]]; then
    echo "SKIP incremental benchmark aa_lcr: $AA_LCR_SKIP_REASON."
    continue
  fi

  echo
  echo "=== Incremental benchmark: $benchmark (infer -> eval -> summary) ==="
  SUMMARY_MARKER="$(mktemp "$TMP_DIR/${benchmark}.summary-marker.XXXXXX")"
  _OPENCOMPASS_LANGUAGE_BENCHMARK="$benchmark" \
    "${cmd[@]}" --reuse "$INCREMENTAL_RUN_ID"

  if [[ "$DRY_RUN" != "1" ]]; then
    STAGE_SUMMARY_DIR="$INCREMENTAL_RUN_DIR/summary/by_benchmark/$benchmark"
    copy_new_summaries "$SUMMARY_MARKER" "$STAGE_SUMMARY_DIR"
    echo "Stage summary: $STAGE_SUMMARY_DIR"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Incremental dry run completed; no evaluation or summary was written."
  exit 0
fi

echo
echo "=== Combined evaluation and summary ==="
COMBINED_MARKER="$(mktemp "$TMP_DIR/combined.summary-marker.XXXXXX")"
_OPENCOMPASS_LANGUAGE_BENCHMARK='' \
  "${cmd[@]}" --mode eval --reuse "$INCREMENTAL_RUN_ID"
COMBINED_SUMMARY_DIR="$INCREMENTAL_RUN_DIR/summary/combined"
copy_new_summaries "$COMBINED_MARKER" "$COMBINED_SUMMARY_DIR"

echo "Incremental evaluation completed."
echo "Per-benchmark summaries: $INCREMENTAL_RUN_DIR/summary/by_benchmark"
echo "Combined summary: $COMBINED_SUMMARY_DIR"
