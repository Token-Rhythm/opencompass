#!/usr/bin/env bash
set -euo pipefail

# Keep multiprocessing Unix socket paths below Linux's AF_UNIX length limit.
RUNTIME_TMPDIR="${OPENCOMPASS_RUNTIME_TMPDIR:-/tmp/pipeline-eval-opencompass}"
mkdir -p "$RUNTIME_TMPDIR"
export TMPDIR="$RUNTIME_TMPDIR" TMP="$RUNTIME_TMPDIR" TEMP="$RUNTIME_TMPDIR"

# Shared launcher for the complete benchmark suite. Generation tasks use an
# OpenAI-compatible vLLM chat endpoint; probability tasks such as INCLUDE use
# the same server's raw completions endpoint with prompt log-probabilities.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BENCHMARK [options]" >&2
  exit 1
fi

BENCHMARK="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="http://47.116.36.105:8080/v1"
MODEL="Qwen3.5-2B"
API_KEY="EMPTY"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/../opencompass-venv/bin/python}"
TOKENIZER_PATH="Qwen/Qwen3.5-2B"
MAX_SEQ_LEN="65536"
MAX_OUT_LEN="4096"
OMIT_MAX_OUT_LEN="0"
TEMPERATURE="1.0"
QUERY_PER_SECOND="64"
BATCH_SIZE="512"
RETRY="1"
TIMEOUT="3600"
STREAM_IDLE_TIMEOUT="600"
MAX_WORKERS="64"
DATASET_WORKERS="1"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-60}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-3}"
PREFLIGHT_BACKOFF="${PREFLIGHT_BACKOFF:-5}"
EXTRA_BODY_JSON='{"top_k":20,"min_p":0.0,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0}'
OPENAI_EXTRA_KWARGS_JSON='{}'
INPUT_TRUNCATION_MODE="none"
SAMPLES="all"
DRY_RUN="0"
REUSE_RUN=""
STREAM_RESPONSES="1"
TEST_RANGES_JSON='{}'
DATASET_KWARGS_JSON='{}'
RUN_MODE="all"
PROTOCOL="chat"
DATASETS_EXPR=""
DATASET_ABBR_SUFFIX=""
SUMMARY_GROUP_SUFFIX=""
DOWNSAMPLING_KEY=""
INFERENCER_MAX_WORKERS_OVERRIDE="0"
AGIEVAL_SETTING="zero-shot"
AGIEVAL_OPTIONS_SET="0"
DOWNSAMPLING_MANIFEST="$REPO_ROOT/opencompass/configs/datasets/downsampling/manifest.json"

case "$BENCHMARK" in
  agieval|agieval_v1_1)
    BENCHMARK_LABEL="AGIEval v1.1"
    DATASET_MODULE="opencompass.configs.datasets.agieval.agieval_v1_1_gen"
    DATASET_VARIABLE="agieval_v1_1_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="agieval_v1_1_summary_groups"
    SUMMARY_ABBRS_EXPR="[]"
    # Resolved after protocol options; an explicit --work-dir takes priority.
    WORK_DIR=""
    ;;
  mmlu_pro)
    BENCHMARK_LABEL="MMLU-Pro"
    DATASET_MODULE="opencompass.configs.datasets.mmlu_pro.mmlu_pro_5shot_cot_gen"
    DATASET_VARIABLE="mmlu_pro_5shot_cot_datasets"
    SUMMARY_IMPORT="from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups"
    SUMMARY_GROUPS_EXPR="mmlu_pro_summary_groups"
    SUMMARY_ABBRS_EXPR="[['mmlu_pro', 'accuracy'], ['mmlu_pro_biology', 'accuracy'], ['mmlu_pro_business', 'accuracy'], ['mmlu_pro_chemistry', 'accuracy'], ['mmlu_pro_computer_science', 'accuracy'], ['mmlu_pro_economics', 'accuracy'], ['mmlu_pro_engineering', 'accuracy'], ['mmlu_pro_health', 'accuracy'], ['mmlu_pro_history', 'accuracy'], ['mmlu_pro_law', 'accuracy'], ['mmlu_pro_math', 'accuracy'], ['mmlu_pro_other', 'accuracy'], ['mmlu_pro_philosophy', 'accuracy'], ['mmlu_pro_physics', 'accuracy'], ['mmlu_pro_psychology', 'accuracy']]"
    WORK_DIR="outputs/mmlu_pro_full_chat_qwen3.5_2b_thinking"
    MAX_OUT_LEN="16384"
    ;;
  ceval)
    BENCHMARK_LABEL="C-Eval"
    DATASET_MODULE="opencompass.configs.datasets.ceval.ceval_gen"
    DATASET_VARIABLE="ceval_datasets"
    SUMMARY_IMPORT="from opencompass.configs.summarizers.groups.ceval import ceval_summary_groups"
    SUMMARY_GROUPS_EXPR="ceval_summary_groups"
    SUMMARY_ABBRS_EXPR="[['ceval', 'naive_average'], ['ceval-hard', 'naive_average'], ['ceval-stem', 'naive_average'], ['ceval-social-science', 'naive_average'], ['ceval-humanities', 'naive_average'], ['ceval-other', 'naive_average']]"
    WORK_DIR="outputs/ceval_full_chat"
    MAX_OUT_LEN="8192"
    ;;
  ceval_evalscope)
    BENCHMARK_LABEL="C-Eval (EvalScope single-context 5-shot)"
    DATASET_MODULE="opencompass.configs.datasets.ceval.ceval_evalscope_gen"
    DATASET_VARIABLE="ceval_evalscope_datasets"
    SUMMARY_IMPORT="from opencompass.configs.summarizers.groups.ceval import ceval_summary_groups"
    SUMMARY_GROUPS_EXPR="ceval_summary_groups"
    SUMMARY_ABBRS_EXPR="[['ceval', 'naive_average'], ['ceval-hard', 'naive_average'], ['ceval-stem', 'naive_average'], ['ceval-social-science', 'naive_average'], ['ceval-humanities', 'naive_average'], ['ceval-other', 'naive_average']]"
    WORK_DIR="outputs/ceval_evalscope_full_chat"
    MAX_OUT_LEN="16384"
    ;;
  supergpqa)
    BENCHMARK_LABEL="SuperGPQA"
    DATASET_MODULE="opencompass.configs.datasets.supergpqa.supergpqa_gen"
    DATASET_VARIABLE="supergpqa_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['supergpqa', 'accuracy'], ['supergpqa', 'hard_accuracy'], ['supergpqa', 'middle_accuracy'], ['supergpqa', 'easy_accuracy']]"
    WORK_DIR="outputs/supergpqa_full_chat"
    MAX_OUT_LEN="16384"
    ;;
  gpqa_diamond)
    BENCHMARK_LABEL="GPQA-Diamond"
    DATASET_MODULE="opencompass.configs.datasets.gpqa.gpqa_gen"
    DATASET_VARIABLE="gpqa_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['GPQA_diamond', 'accuracy']]"
    WORK_DIR="outputs/gpqa_diamond_full_chat"
    MAX_OUT_LEN="16384"
    ;;
  ifeval)
    BENCHMARK_LABEL="IFEval"
    DATASET_MODULE="opencompass.configs.datasets.IFEval.IFEval_gen"
    DATASET_VARIABLE="ifeval_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['IFEval', 'Prompt-level-strict-accuracy'], ['IFEval', 'Inst-level-strict-accuracy'], ['IFEval', 'Prompt-level-loose-accuracy'], ['IFEval', 'Inst-level-loose-accuracy']]"
    WORK_DIR="outputs/ifeval_full_chat"
    MAX_OUT_LEN="8192"
    ;;
  ifbench)
    BENCHMARK_LABEL="IFBench"
    DATASET_MODULE="opencompass.configs.datasets.IFBench.IFBench_gen"
    DATASET_VARIABLE="ifbench_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['IFBench', 'score'], ['IFBench', 'Prompt-level-strict-accuracy'], ['IFBench', 'Inst-level-strict-accuracy'], ['IFBench', 'Prompt-level-loose-accuracy'], ['IFBench', 'Inst-level-loose-accuracy'], ['IFBench', 'average_4_metrics']]"
    WORK_DIR="outputs/ifbench_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  longbenchv2)
    BENCHMARK_LABEL="LongBench v2"
    DATASET_MODULE="opencompass.configs.datasets.longbenchv2.longbenchv2_gen"
    DATASET_VARIABLE="LongBenchv2_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['LongBenchv2', 'accuracy'], ['LongBenchv2', 'accuracy_easy'], ['LongBenchv2', 'accuracy_hard'], ['LongBenchv2', 'accuracy_short'], ['LongBenchv2', 'accuracy_medium'], ['LongBenchv2', 'accuracy_long']]"
    WORK_DIR="outputs/longbenchv2_full_chat"
    MAX_OUT_LEN="8192"
    INPUT_TRUNCATION_MODE="token_mid"
    ;;
  aa_lcr)
    BENCHMARK_LABEL="AA-LCR"
    DATASET_MODULE="opencompass.configs.datasets.aa_lcr.aa_lcr_gen"
    DATASET_VARIABLE="aa_lcr_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['aa_lcr', 'accuracy']]"
    WORK_DIR="outputs/aa_lcr_full_chat"
    MAX_OUT_LEN="32768"
    INPUT_TRUNCATION_MODE="mid"
    ;;
  aime2024)
    BENCHMARK_LABEL="AIME 2024"
    DATASET_MODULE="opencompass.configs.datasets.aime2024.aime2024_gen"
    DATASET_VARIABLE="aime2024_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['aime2024', 'accuracy']]"
    WORK_DIR="outputs/aime2024_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  aime2025)
    BENCHMARK_LABEL="AIME 2025"
    DATASET_MODULE="opencompass.configs.datasets.aime2025.aime2025_gen"
    DATASET_VARIABLE="aime2025_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['aime2025', 'accuracy']]"
    WORK_DIR="outputs/aime2025_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  aime2026)
    BENCHMARK_LABEL="AIME 2026"
    DATASET_MODULE="opencompass.configs.datasets.aime2026.aime2026_gen"
    DATASET_VARIABLE="aime2026_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['aime2026', 'accuracy']]"
    WORK_DIR="outputs/aime2026_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  hmmt2026)
    BENCHMARK_LABEL="HMMT February 2026"
    DATASET_MODULE="opencompass.configs.datasets.hmmt2026.hmmt2026_gen"
    DATASET_VARIABLE="hmmt2026_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['hmmt2026', 'accuracy']]"
    WORK_DIR="outputs/hmmt_feb_2026_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  scicode)
    BENCHMARK_LABEL="SciCode official with background"
    DATASET_MODULE="opencompass.configs.datasets.scicode.scicode_official_wbg_gen"
    DATASET_VARIABLE="SciCode_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['SciCode_official_with_background_sandboxed', 'accuracy'], ['SciCode_official_with_background_sandboxed', 'sub_accuracy']]"
    WORK_DIR="outputs/scicode_official_wbg_full_chat"
    MAX_OUT_LEN="4096"
    OMIT_MAX_OUT_LEN="1"
    INFERENCER_MAX_WORKERS_OVERRIDE="1"
    ;;
  livecodebench)
    BENCHMARK_LABEL="LiveCodeBench v6 Code Generation"
    DATASET_MODULE="opencompass.configs.datasets.livecodebench.livecodebench_v6_codegen"
    DATASET_VARIABLE="livecodebench_v6_codegen_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['livecodebench_v6_codegen', 'pass@1']]"
    WORK_DIR="outputs/livecodebench_v6_codegen_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  humaneval)
    BENCHMARK_LABEL="HumanEval"
    DATASET_MODULE="opencompass.configs.datasets.humaneval.humaneval_gen"
    DATASET_VARIABLE="humaneval_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['openai_humaneval', 'humaneval_pass@1']]"
    WORK_DIR="outputs/humaneval_full_chat"
    MAX_OUT_LEN="32768"
    ;;
  mmmlu|mmmlu_downsampling)
    BENCHMARK_LABEL="OpenAI MMMLU"
    DATASET_MODULE="opencompass.configs.datasets.mmmlu.mmmlu_gen"
    DATASET_VARIABLE="mmmlu_datasets"
    SUMMARY_IMPORT="from opencompass.configs.summarizers.groups.mmmlu import mmmlu_summary_groups"
    SUMMARY_GROUPS_EXPR="mmmlu_summary_groups"
    SUMMARY_ABBRS_EXPR="['mmmlu', 'openai_mmmlu_AR-XY', 'openai_mmmlu_BN-BD', 'openai_mmmlu_DE-DE', 'openai_mmmlu_ES-LA', 'openai_mmmlu_FR-FR', 'openai_mmmlu_HI-IN', 'openai_mmmlu_ID-ID', 'openai_mmmlu_IT-IT', 'openai_mmmlu_JA-JP', 'openai_mmmlu_KO-KR', 'openai_mmmlu_PT-BR', 'openai_mmmlu_SW-KE', 'openai_mmmlu_YO-NG', 'openai_mmmlu_ZH-CN']"
    WORK_DIR="outputs/mmmlu_full_chat"
    MAX_OUT_LEN="16384"
    if [[ "$BENCHMARK" == "mmmlu_downsampling" ]]; then
      BENCHMARK_LABEL="OpenAI MMMLU (stratified downsampling)"
      WORK_DIR="outputs/mmmlu_downsampling"
      DOWNSAMPLING_KEY="mmmlu"
      DATASET_ABBR_SUFFIX="_downsampling"
      SUMMARY_GROUP_SUFFIX="_downsampling"
    fi
    ;;
  mmlu_prox|mmlu_prox_downsampling)
    BENCHMARK_LABEL="MMLU-ProX Full 5-shot CoT"
    DATASET_MODULE="opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen"
    DATASET_VARIABLE="mmlu_prox_5shot_datasets"
    SUMMARY_IMPORT="from opencompass.configs.summarizers.groups.mmlu_prox import mmlu_prox_summary_groups"
    SUMMARY_GROUPS_EXPR="mmlu_prox_summary_groups"
    SUMMARY_ABBRS_EXPR="['mmlu_prox']"
    WORK_DIR="outputs/mmlu_prox_full_5shot_chat"
    MAX_OUT_LEN="16384"
    if [[ "$BENCHMARK" == "mmlu_prox_downsampling" ]]; then
      BENCHMARK_LABEL="MMLU-ProX Full 5-shot CoT (stratified downsampling)"
      WORK_DIR="outputs/mmlu_prox_downsampling"
      DOWNSAMPLING_KEY="mmlu_prox"
      DATASET_ABBR_SUFFIX="_downsampling"
      SUMMARY_GROUP_SUFFIX="_downsampling"
    fi
    ;;
  global_piqa|global_piqa_downsampling)
    BENCHMARK_LABEL="Global PIQA"
    DATASET_MODULE="opencompass.configs.datasets.global_piqa.global_piqa_generation"
    DATASET_VARIABLE="global_piqa_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['global_piqa_generation', 'accuracy'], ['global_piqa_generation', 'nonparallel_accuracy'], ['global_piqa_generation', 'parallel_accuracy']]"
    WORK_DIR="outputs/global_piqa_full_chat"
    MAX_OUT_LEN="16384"
    if [[ "$BENCHMARK" == "global_piqa_downsampling" ]]; then
      BENCHMARK_LABEL="Global PIQA (stratified downsampling)"
      WORK_DIR="outputs/global_piqa_downsampling"
      DOWNSAMPLING_KEY="global_piqa"
      DATASET_ABBR_SUFFIX="_downsampling"
      SUMMARY_GROUP_SUFFIX="_downsampling"
    fi
    ;;
  mmlu_redux|mmlu_redux_downsampling)
    BENCHMARK_LABEL="MMLU-Redux 2.0"
    DATASET_MODULE="opencompass.configs.datasets.mmlu_redux.mmlu_redux_gen"
    DATASET_VARIABLE="mmlu_redux_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['mmlu_redux_full_chat', 'accuracy']]"
    WORK_DIR="outputs/mmlu_redux_full_chat"
    TIMEOUT="7200"
    DATASET_ABBR_SUFFIX="_full_chat"
    if [[ "$BENCHMARK" == "mmlu_redux_downsampling" ]]; then
      BENCHMARK_LABEL="MMLU-Redux 2.0 (stratified downsampling)"
      SUMMARY_ABBRS_EXPR="[['mmlu_redux', 'accuracy']]"
      WORK_DIR="outputs/mmlu_redux_downsampling"
      DOWNSAMPLING_KEY="mmlu_redux"
      DATASET_ABBR_SUFFIX="_downsampling"
      SUMMARY_GROUP_SUFFIX="_downsampling"
    fi
    ;;
  hmmt_feb_2025)
    BENCHMARK_LABEL="HMMT February 2025"
    DATASET_MODULE="opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen"
    DATASET_VARIABLE="hmmt_2025_datasets"
    DATASETS_EXPR="[hmmt_2025_datasets[0]]"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['hmmt_feb_2025_full_chat', 'accuracy']]"
    WORK_DIR="outputs/hmmt_feb_2025_full_chat"
    MAX_OUT_LEN="32768"
    TIMEOUT="7200"
    DATASET_ABBR_SUFFIX="_full_chat"
    ;;
  hmmt_nov_2025)
    BENCHMARK_LABEL="HMMT November 2025"
    DATASET_MODULE="opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen"
    DATASET_VARIABLE="hmmt_2025_datasets"
    DATASETS_EXPR="[hmmt_2025_datasets[1]]"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['hmmt_nov_2025_full_chat', 'accuracy']]"
    WORK_DIR="outputs/hmmt_nov_2025_full_chat"
    MAX_OUT_LEN="32768"
    TIMEOUT="7200"
    DATASET_ABBR_SUFFIX="_full_chat"
    ;;
  multichallenge)
    BENCHMARK_LABEL="MultiChallenge"
    DATASET_MODULE="opencompass.configs.datasets.multichallenge.multichallenge_gen"
    DATASET_VARIABLE="multichallenge_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[['multichallenge', 'overall_score']]"
    WORK_DIR="outputs/multichallenge_full_chat"
    MAX_OUT_LEN="32768"
    TIMEOUT="7200"
    ;;
  include|include_downsampling)
    BENCHMARK_LABEL="INCLUDE base-44"
    DATASET_MODULE="opencompass.configs.datasets.include.include_base_44_0shot_ppl"
    DATASET_VARIABLE="include_datasets"
    SUMMARY_IMPORT=""
    SUMMARY_GROUPS_EXPR="[]"
    SUMMARY_ABBRS_EXPR="[]"
    WORK_DIR="outputs/include_base_44"
    PROTOCOL="logprob"
    DATASET_ABBR_SUFFIX="_full_chat"
    if [[ "$BENCHMARK" == "include_downsampling" ]]; then
      BENCHMARK_LABEL="INCLUDE base-44 (stratified downsampling)"
      WORK_DIR="outputs/include_downsampling"
      DOWNSAMPLING_KEY="include"
      DATASET_ABBR_SUFFIX="_downsampling"
      SUMMARY_GROUP_SUFFIX="_downsampling"
    fi
    # Generation sampling does not participate in prompt-logprob scoring.
    TEMPERATURE="0"
    EXTRA_BODY_JSON='{}'
    OPENAI_EXTRA_KWARGS_JSON='{}'
    STREAM_RESPONSES="0"
    ;;
  *)
    echo "Unsupported benchmark: $BENCHMARK" >&2
    echo "Expected one of: agieval, mmlu_pro, ceval, ceval_evalscope, supergpqa, gpqa_diamond, ifeval, ifbench, longbenchv2, aa_lcr, aime2024, aime2025, aime2026, hmmt2026, scicode, livecodebench, humaneval, mmmlu, mmlu_prox, global_piqa, mmlu_redux, hmmt_feb_2025, hmmt_nov_2025, multichallenge, include" >&2
    exit 1
    ;;
esac

usage() {
  cat <<EOF
Usage:
  bash script/run_posttrain_objective_benchmark.sh ${BENCHMARK} [options]

Runs the full ${BENCHMARK_LABEL} benchmark through the protocol selected by
its audited dataset configuration. Defaults are tuned for a vLLM
service: 65536 context, 4096 output tokens, batches of 512 prompts and 64 API
worker threads. Dataset partitions stay in one local task process unless
--dataset-workers is explicitly increased for many-small-subset benchmarks.

Options:
  --base-url URL              Default: $BASE_URL
  --model NAME               Default: $MODEL
  --api-key KEY              Default: $API_KEY
  --python PATH              Default: ../opencompass-venv/bin/python
  --tokenizer-path PATH      Tokenizer used for context accounting; default: $TOKENIZER_PATH
  --work-dir PATH            Default: $WORK_DIR
  --max-seq-len N            Default: $MAX_SEQ_LEN
  --max-out-len N            Default: $MAX_OUT_LEN
  --omit-max-out-len       Omit max_tokens from the inference request
  --temperature FLOAT        Default: $TEMPERATURE
  --query-per-second N       Default: $QUERY_PER_SECOND
  --batch-size N             Prompts queued per inference batch; default: $BATCH_SIZE
  --max-workers N            Concurrent API request threads; default: $MAX_WORKERS
  --dataset-workers N        Concurrent OpenCompass dataset partitions;
                              default: $DATASET_WORKERS. Total API concurrency
                              can reach N times --max-workers, so raise only
                              when individual subsets are small.
  --retry N                  Default: $RETRY
  --timeout SECONDS          Default: $TIMEOUT
  --stream-idle-timeout SECONDS
                              Maximum idle time between streamed response
                              chunks; received text is preserved if a proxy
                              omits the terminal SSE event. Default:
                              $STREAM_IDLE_TIMEOUT
  --samples N|all            Optional per-dataset smoke-test limit
  --test-ranges-json JSON    Map dataset abbrs to explicit zero-based index lists;
                              when non-empty, only mapped datasets are run
  --dataset-kwargs-json JSON Extra keyword arguments passed to every selected
                              dataset loader; default: {}
  --run-mode MODE             all, infer, eval, or viz; default: $RUN_MODE
  --agieval-setting MODE      AGIEval only: zero-shot (default),
                              or zero-shot-CoT
  --extra-body-json JSON     Default: $EXTRA_BODY_JSON
  --openai-extra-kwargs-json JSON
                              Default: $OPENAI_EXTRA_KWARGS_JSON
  --input-truncation-mode MODE
                              none, front, mid, rear, or token_mid; default: $INPUT_TRUNCATION_MODE
  --reuse RUN_ID             Resume an existing timestamp under --work-dir
  --stream-responses         Stream chat-generation responses
  --no-stream-responses      Disable chat streaming
  --dry-run                  Build and partition tasks without inference
  -h, --help                 Show this help

Protocol note:
  INCLUDE ignores generation sampling/output options. It always uses
  /v1/completions with temperature=0, max_tokens=1, return_token_ids and
  prompt_logprobs for official continuation-likelihood scoring.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agieval-setting)
      [[ $# -ge 2 && "$2" != --* ]] || {
        echo "$1 requires a value" >&2; exit 1;
      }
      AGIEVAL_OPTIONS_SET="1"
      AGIEVAL_SETTING="$2"
      shift 2
      ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-out-len) MAX_OUT_LEN="$2"; shift 2 ;;
    --omit-max-out-len) OMIT_MAX_OUT_LEN="1"; shift ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --query-per-second) QUERY_PER_SECOND="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --retry) RETRY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --stream-idle-timeout) STREAM_IDLE_TIMEOUT="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --dataset-workers) DATASET_WORKERS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --test-ranges-json) TEST_RANGES_JSON="$2"; shift 2 ;;
    --dataset-kwargs-json) DATASET_KWARGS_JSON="$2"; shift 2 ;;
    --run-mode) RUN_MODE="$2"; shift 2 ;;
    --extra-body-json) EXTRA_BODY_JSON="$2"; shift 2 ;;
    --openai-extra-kwargs-json) OPENAI_EXTRA_KWARGS_JSON="$2"; shift 2 ;;
    --input-truncation-mode) INPUT_TRUNCATION_MODE="$2"; shift 2 ;;
    --reuse) REUSE_RUN="$2"; shift 2 ;;
    --stream-responses) STREAM_RESPONSES="1"; shift ;;
    --no-stream-responses) STREAM_RESPONSES="0"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$BENCHMARK" == "agieval" || "$BENCHMARK" == "agieval_v1_1" ]]; then
  case "$AGIEVAL_SETTING" in
    zero-shot)
      AGIEVAL_MODE_SUFFIX="zero_shot_chat"
      ;;
    zero-shot-CoT)
      DATASET_MODULE="opencompass.configs.datasets.agieval.agieval_v1_1_zeroshot_cot_gen"
      AGIEVAL_MODE_SUFFIX="zero_shot_cot_chat"
      ;;
    *)
      echo "--agieval-setting must be zero-shot or zero-shot-CoT" >&2
      exit 1
      ;;
  esac
  AGIEVAL_SUMMARY_ROOT="agieval_v1_1_${AGIEVAL_MODE_SUFFIX}"
  SUMMARY_IMPORT="from ${DATASET_MODULE} import agieval_v1_1_summary_groups"
  SUMMARY_ABBRS_EXPR="['${AGIEVAL_SUMMARY_ROOT}', '${AGIEVAL_SUMMARY_ROOT}_en', '${AGIEVAL_SUMMARY_ROOT}_zh', '${AGIEVAL_SUMMARY_ROOT}_cloze']"
  WORK_DIR="${WORK_DIR:-outputs/${AGIEVAL_SUMMARY_ROOT}}"
  BENCHMARK_LABEL="AGIEval v1.1 (${AGIEVAL_SETTING})"
elif [[ "$AGIEVAL_OPTIONS_SET" == "1" ]]; then
  echo "--agieval-* options require the agieval benchmark" >&2
  exit 1
fi

for numeric_value in MAX_SEQ_LEN MAX_OUT_LEN BATCH_SIZE RETRY TIMEOUT STREAM_IDLE_TIMEOUT MAX_WORKERS DATASET_WORKERS PREFLIGHT_TIMEOUT PREFLIGHT_ATTEMPTS PREFLIGHT_BACKOFF; do
  value="${!numeric_value}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric_value must be a positive integer, got: $value" >&2
    exit 1
  fi
done
if [[ "$SAMPLES" != "all" && ! "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--samples must be a positive integer or 'all', got: $SAMPLES" >&2
  exit 1
fi
if [[ ! "$INPUT_TRUNCATION_MODE" =~ ^(none|front|mid|rear|token_mid)$ ]]; then
  echo "--input-truncation-mode must be none, front, mid, rear, or token_mid" >&2
  exit 1
fi
if [[ ! "$RUN_MODE" =~ ^(all|infer|eval|viz)$ ]]; then
  echo "--run-mode must be all, infer, eval, or viz" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"
NORMALIZED_BASE_URL="${BASE_URL%/}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/chat/completions}"
NORMALIZED_BASE_URL="${NORMALIZED_BASE_URL%/completions}"

if [[ -n "${OPENCOMPASS_SERVED_MAX_SEQ_LEN_OVERRIDE:-}" ]]; then
  [[ "$OPENCOMPASS_SERVED_MAX_SEQ_LEN_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || {
    echo "OPENCOMPASS_SERVED_MAX_SEQ_LEN_OVERRIDE must be a positive integer" >&2
    exit 1
  }
  SERVED_MAX_SEQ_LEN="$OPENCOMPASS_SERVED_MAX_SEQ_LEN_OVERRIDE"
else
  SERVED_MAX_SEQ_LEN="$($PYTHON_BIN - "$NORMALIZED_BASE_URL" "$API_KEY" "$MODEL" "$PREFLIGHT_TIMEOUT" "$PREFLIGHT_ATTEMPTS" "$PREFLIGHT_BACKOFF" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

base_url, api_key, model, timeout_text, attempts_text, backoff_text = sys.argv[1:]
timeout = int(timeout_text)
attempts = int(attempts_text)
backoff = int(backoff_text)

for attempt in range(1, attempts + 1):
    request = urllib.request.Request(
        base_url.rstrip('/') + '/models',
        headers={'Authorization': f'Bearer {api_key}'},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        match = next((item for item in payload.get('data', [])
                      if item.get('id') == model), None)
        if match is None:
            raise RuntimeError(f'model {model!r} is not listed by /v1/models')
        max_model_len = match.get('max_model_len')
        if not isinstance(max_model_len, int):
            raise RuntimeError('/v1/models did not return max_model_len')
        print(max_model_len)
        break
    except (urllib.error.HTTPError, RuntimeError) as error:
        print(f'ERROR:{type(error).__name__}: {error}')
        break
    except Exception as error:
        if attempt == attempts:
            print(f'ERROR:{type(error).__name__}: {error} after {attempts} attempts')
            break
        delay = backoff * attempt
        print(
            f'preflight attempt {attempt}/{attempts} failed: '
            f'{type(error).__name__}: {error}; retrying in {delay}s',
            file=sys.stderr,
        )
        time.sleep(delay)
PY
)"
fi
if [[ "$SERVED_MAX_SEQ_LEN" == ERROR:* ]]; then
  echo "vLLM endpoint preflight failed at ${NORMALIZED_BASE_URL}:" >&2
  echo "${SERVED_MAX_SEQ_LEN#ERROR:}" >&2
  exit 1
fi
if (( MAX_SEQ_LEN > SERVED_MAX_SEQ_LEN )); then
  echo "Configured max_seq_len=$MAX_SEQ_LEN exceeds server max_model_len=$SERVED_MAX_SEQ_LEN." >&2
  exit 1
fi

json_to_python() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if value is not None and not isinstance(value, dict):
    raise SystemExit('expected a JSON object')
print(repr(value or None))
PY
}

EXTRA_BODY_PY="$(json_to_python "$EXTRA_BODY_JSON")"
OPENAI_EXTRA_KWARGS_PY="$(json_to_python "$OPENAI_EXTRA_KWARGS_JSON")"
if [[ -n "$DOWNSAMPLING_KEY" ]]; then
  [[ "$TEST_RANGES_JSON" == "{}" ]] || {
    echo "downsampling benchmarks cannot be combined with --test-ranges-json" >&2
    exit 1
  }
  [[ -f "$DOWNSAMPLING_MANIFEST" ]] || {
    echo "Downsampling manifest is missing: $DOWNSAMPLING_MANIFEST" >&2
    exit 1
  }
  TEST_RANGES_JSON="$($PYTHON_BIN - "$DOWNSAMPLING_MANIFEST" "$DOWNSAMPLING_KEY" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding='utf-8'))
benchmark = manifest['benchmarks'][sys.argv[2]]
ranges = {abbr: item['indices'] for abbr, item in benchmark['datasets'].items()}
if sum(map(len, ranges.values())) != benchmark['sample_total']:
    raise SystemExit('downsampling manifest sample_total does not match indices')
print(json.dumps(ranges, separators=(',', ':')))
PY
)"
  mkdir -p "$WORK_DIR"
  cp "$DOWNSAMPLING_MANIFEST" "$WORK_DIR/downsampling_manifest.json"
fi
TEST_RANGES_PY="$($PYTHON_BIN - "$TEST_RANGES_JSON" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit('--test-ranges-json must be a JSON object')
for abbr, indices in value.items():
    if not isinstance(abbr, str) or not isinstance(indices, list):
        raise SystemExit('--test-ranges-json values must be index lists')
    if (not indices
            or any(not isinstance(index, int) or isinstance(index, bool)
            or index < 0 for index in indices)
            or len(indices) != len(set(indices))):
        raise SystemExit('explicit index lists must be non-empty and contain '
                         'unique non-negative integers')
print(repr(value))
PY
)"
DATASET_KWARGS_PY="$($PYTHON_BIN - "$DATASET_KWARGS_JSON" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, dict) or any(not isinstance(key, str)
                                      for key in value):
    raise SystemExit('--dataset-kwargs-json must be a JSON object with '
                     'string keys')
print(repr(value))
PY
)"
if [[ "$SAMPLES" != "all" && "$TEST_RANGES_PY" != "{}" ]]; then
  echo "--samples and --test-ranges-json cannot be used together" >&2
  exit 1
fi
if [[ -z "$DATASETS_EXPR" ]]; then
  DATASETS_EXPR="$DATASET_VARIABLE"
fi
if [[ "$STREAM_RESPONSES" == "1" ]]; then
  STREAM_RESPONSES_PY="True"
else
  STREAM_RESPONSES_PY="False"
fi
if [[ "$INFERENCER_MAX_WORKERS_OVERRIDE" == "1" ]]; then
  INFERENCER_MAX_WORKERS_OVERRIDE_PY="True"
else
  INFERENCER_MAX_WORKERS_OVERRIDE_PY="False"
fi
MAX_OUT_LEN_PY="$MAX_OUT_LEN"
if [[ "$OMIT_MAX_OUT_LEN" == "1" ]]; then
  MAX_OUT_LEN_PY="None"
fi
BASE_URL_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$NORMALIZED_BASE_URL")"
API_KEY_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$API_KEY")"
MODEL_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$MODEL")"
TOKENIZER_PATH_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$TOKENIZER_PATH")"
INPUT_TRUNCATION_MODE_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$INPUT_TRUNCATION_MODE")"
PROTOCOL_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$PROTOCOL")"
DATASET_ABBR_SUFFIX_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$DATASET_ABBR_SUFFIX")"
SUMMARY_GROUP_SUFFIX_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$SUMMARY_GROUP_SUFFIX")"
ABBR="$(printf '%s' "$MODEL" | sed 's/[^A-Za-z0-9_.-]/_/g')"
ABBR_PY="$($PYTHON_BIN -c 'import sys; print(repr(sys.argv[1]))' "$ABBR")"
if [[ "$SAMPLES" == "all" ]]; then
  TEST_RANGE_PY="None"
else
  TEST_RANGE_PY="$($PYTHON_BIN -c 'import sys; print(repr("[:" + sys.argv[1] + "]"))' "$SAMPLES")"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opencompass-${BENCHMARK}.XXXXXX")"
CONFIG_PATH="$TMP_DIR/${BENCHMARK}_${PROTOCOL}.py"

cat > "$CONFIG_PATH" <<PY
from mmengine.config import read_base
from opencompass.models import VLLMOpenAIAPI

with read_base():
    from ${DATASET_MODULE} import ${DATASET_VARIABLE}
    ${SUMMARY_IMPORT}

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

protocol = ${PROTOCOL_PY}
model_cfg = dict(
    type=VLLMOpenAIAPI,
    abbr=${ABBR_PY} + ('-completions' if protocol == 'logprob' else '-chat'),
    summarizer_abbr=${ABBR_PY},
    path=${MODEL_PY},
    tokenizer_path=${TOKENIZER_PATH_PY},
    key=${API_KEY_PY},
    openai_api_base=${BASE_URL_PY},
    max_seq_len=${MAX_SEQ_LEN},
    mode=${INPUT_TRUNCATION_MODE_PY},
    max_out_len=${MAX_OUT_LEN_PY},
    query_per_second=${QUERY_PER_SECOND},
    batch_size=${BATCH_SIZE},
    retry=${RETRY},
    timeout=${TIMEOUT},
    max_workers=${MAX_WORKERS},
)
if protocol == 'logprob':
    # LLInferencer calls get_loglikelihood(), which fixes temperature=0 and
    # requests return_token_ids/prompt_logprobs from /v1/completions.
    model_cfg.update(
        generation_endpoint='completions',
        meta_template=None,
        temperature=0,
        extra_body=None,
        openai_extra_kwargs=None,
        completion_extra_body={},
        stream_chat=False,
    )
else:
    model_cfg.update(
        generation_endpoint='chat',
        stream_chat=${STREAM_RESPONSES_PY},
        stream_idle_timeout=${STREAM_IDLE_TIMEOUT},
        meta_template=api_meta_template,
        temperature=${TEMPERATURE},
        extra_body=${EXTRA_BODY_PY},
        openai_extra_kwargs=${OPENAI_EXTRA_KWARGS_PY},
    )
models = [model_cfg]

datasets = ${DATASETS_EXPR}
test_range = ${TEST_RANGE_PY}
test_ranges = ${TEST_RANGES_PY}
dataset_kwargs = ${DATASET_KWARGS_PY}
dataset_abbr_suffix = ${DATASET_ABBR_SUFFIX_PY}
summary_group_suffix = ${SUMMARY_GROUP_SUFFIX_PY}
selected_datasets = []
for dataset in datasets:
    if dataset.get('type') is not None and dataset.get('setting_name') in (
            'zero-shot', 'zero-shot-CoT') and dataset.get(
                'abbr', '').startswith('agieval_v1_1_'):
        for key in ('setting_name', 'chat_mode'):
            if key in dataset_kwargs and dataset_kwargs[key] != dataset[key]:
                raise ValueError(
                    'Use --agieval-setting to change '
                    'the protocol; dataset-kwargs must not contradict it.')
    dataset.update(dataset_kwargs)
    dataset_abbr = dataset['abbr']
    if test_ranges and dataset_abbr not in test_ranges:
        continue
    if test_ranges:
        dataset['reader_cfg']['test_range'] = test_ranges[dataset_abbr]
    if test_range is not None:
        dataset['reader_cfg']['test_range'] = test_range
    if dataset_abbr_suffix and not dataset_abbr.endswith(dataset_abbr_suffix):
        dataset['abbr'] = dataset_abbr + dataset_abbr_suffix
    if protocol != 'logprob':
        inferencer = dataset['infer_cfg']['inferencer']
        inferencer['max_seq_len'] = ${MAX_SEQ_LEN}
        inferencer['max_out_len'] = ${MAX_OUT_LEN_PY}
        if ${INFERENCER_MAX_WORKERS_OVERRIDE_PY}:
            # Official SciCode parallelizes independent problems here while
            # keeping each problem's dependent sub-steps sequential.
            inferencer['max_infer_workers'] = ${MAX_WORKERS}
    evaluator_dataset = dataset['eval_cfg']['evaluator'].get('dataset_cfg')
    if evaluator_dataset is not None:
        if test_ranges:
            evaluator_dataset['reader_cfg']['test_range'] = test_ranges[
                dataset_abbr]
        if test_range is not None:
            evaluator_dataset['reader_cfg']['test_range'] = test_range
    selected_datasets.append(dataset)
datasets = selected_datasets

summary_dataset_abbrs = ${SUMMARY_ABBRS_EXPR}
summary_groups = ${SUMMARY_GROUPS_EXPR}
if summary_group_suffix:
    renamed_abbrs = []
    for entry in summary_dataset_abbrs:
        if isinstance(entry, str):
            renamed_abbrs.append(entry + summary_group_suffix)
        else:
            renamed_entry = list(entry)
            renamed_entry[0] = renamed_entry[0] + summary_group_suffix
            renamed_abbrs.append(renamed_entry)
    summary_dataset_abbrs = renamed_abbrs
    for group in summary_groups:
        group['name'] = group['name'] + summary_group_suffix
        group['subsets'] = [
            subset + summary_group_suffix for subset in group['subsets']
        ]
        if isinstance(group.get('weights'), dict):
            group['weights'] = {
                key + summary_group_suffix: value
                for key, value in group['weights'].items()
            }

if protocol == 'logprob':
    include_subsets = [dataset['abbr'] for dataset in datasets]
    include_group_name = (
        'include_downsampling'
        if summary_group_suffix else 'include_base_44'
    )
    summarizer = dict(
        dataset_abbrs=[include_group_name, *include_subsets],
        summary_groups=[dict(name=include_group_name,
                             subsets=include_subsets)],
    )
else:
    summarizer = dict(
        dataset_abbrs=summary_dataset_abbrs,
        summary_groups=summary_groups,
    )
del protocol
del dataset_abbr_suffix
del summary_group_suffix
PY

cmd=(
  "$PYTHON_BIN" run.py
  "$CONFIG_PATH"
  --work-dir "$WORK_DIR"
  # API concurrency is controlled independently inside every dataset task.
  # Keep this at one for large subsets; many-small-subset benchmarks can opt
  # into more partitions with --dataset-workers.
  --max-num-workers "$DATASET_WORKERS"
)
if [[ "$RUN_MODE" != "all" ]]; then
  cmd+=(--mode "$RUN_MODE")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ -n "$REUSE_RUN" ]]; then
  cmd+=(--reuse "$REUSE_RUN")
fi

echo "Benchmark: $BENCHMARK_LABEL"
echo "OpenCompass config: $CONFIG_PATH"
echo "Work directory: $WORK_DIR"
if [[ "$PROTOCOL" == "logprob" ]]; then
  echo "Endpoint: $NORMALIZED_BASE_URL/completions (prompt logprobs)"
else
  echo "Endpoint: $NORMALIZED_BASE_URL/chat/completions"
fi
echo "Model: $MODEL"
echo "Samples per dataset: $SAMPLES"
echo "Explicit per-dataset ranges: $TEST_RANGES_JSON"
echo "Dataset loader kwargs: $DATASET_KWARGS_JSON"
if [[ -n "$DOWNSAMPLING_KEY" ]]; then
  echo "Downsampling benchmark: ${DOWNSAMPLING_KEY}_downsampling"
  echo "Downsampling manifest: $DOWNSAMPLING_MANIFEST"
fi
if [[ "$OMIT_MAX_OUT_LEN" == "1" ]]; then
  echo "max_seq_len/max_out_len: $MAX_SEQ_LEN/omitted"
else
  echo "max_seq_len/max_out_len: $MAX_SEQ_LEN/$MAX_OUT_LEN"
fi
echo "batch_size/max_workers: $BATCH_SIZE/$MAX_WORKERS"
echo "OpenCompass dataset workers: $DATASET_WORKERS"
echo "query_per_second/retry/timeout: $QUERY_PER_SECOND/$RETRY/$TIMEOUT"
echo "stream_idle_timeout: $STREAM_IDLE_TIMEOUT"
echo "stream_responses: $STREAM_RESPONSES"
echo "run_mode: $RUN_MODE"
exec "${cmd[@]}"
