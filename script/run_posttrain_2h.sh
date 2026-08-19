#!/usr/bin/env bash
set -euo pipefail

# Two-stage post-training regression launcher.
#
# Stage 1 runs short-context representative benchmarks at high concurrency.
# Stage 2 runs long-context benchmarks at a deliberately lower concurrency.
# Every benchmark is launched separately, so each one is scored before the
# next starts. A hard wall-clock budget stops unfinished work without deleting
# its OpenCompass run directory; rerun with the same --run-id to resume it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME=""
BASE_URL=""
API_KEY="EMPTY"
TOKENIZER_PATH=""
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/../opencompass-venv/bin/python}"
OUTPUT_ROOT="outputs/posttrain_2h"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"

# The order is intentional: return a small math score first, then instruction
# following, then the larger Chinese knowledge benchmark.
SHORT_BENCHMARKS="aime_2025,ifbench,ceval"
LONG_BENCHMARKS="longbench_v2"
SHORT_MAX_SEQ_LEN=65536
LONG_MAX_SEQ_LEN=262144
MAX_OUT_LEN=32768
TEMPERATURE=1.0
EXTRA_BODY_JSON='{"top_k":20,"min_p":0.0,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0}'

SHORT_BATCH_SIZE=1024
SHORT_WORKERS=512
SHORT_QPS=128
# C-Eval consists of many small subject partitions. Running 32 of them in
# parallel keeps both 256-sequence OpenCompass endpoints fed without changing
# the 512 per-task worker cap. Other default short benchmarks use one
# partition.
CEVAL_DATASET_WORKERS=32
SHORT_STREAM_IDLE_TIMEOUT=600
LONG_BATCH_SIZE=32
LONG_WORKERS=32
LONG_QPS=8
LONG_STREAM_IDLE_TIMEOUT=1200
REQUEST_TIMEOUT=7200
RETRY=1
LONG_SAMPLES=all

TIME_BUDGET_SECONDS=7200
# Reserve 70 minutes for full LongBench v2. The short suite is intentionally
# capped at 50 minutes; C-Eval partition parallelism is what makes this target
# plausible on a dedicated two-replica 2B/4B/5B service.
SHORT_BUDGET_SECONDS=3000
TIMEOUT_KILL_AFTER_SECONDS=120
METRICS_URL=""
MONITOR_INTERVAL_SECONDS=60
MONITOR_METRICS=1
DRY_RUN=0
SKIP_SHORT=0
SKIP_LONG=0

usage() {
  cat <<'EOF'
Usage:
  bash script/run_posttrain_2h.sh --model NAME --base-url URL [options]

Default two-stage plan:
  1. aime_2025 -> ifbench -> ceval
     65536 context, 32768 output tokens, 512 workers, QPS 128
  2. longbench_v2 (all 503 samples)
     262144 context, 32768 output tokens, 32 workers, QPS 8

The total wall-clock budget defaults to 7200 seconds. The short stage gets at
most 3000 seconds, reserving 4200 seconds for the long stage. A timeout
does not delete partial predictions. Resume them with the same --run-id.

Required:
  --model NAME                Served model name
  --base-url URL              OpenAI-compatible URL ending in /v1

Common options:
  --api-key KEY               Default: EMPTY
  --tokenizer-path PATH       Default: --model
  --python PATH               OpenCompass environment Python
  --output-root DIR           Default: outputs/posttrain_2h
  --run-id ID                 Reuse this ID to resume missing work
  --temperature FLOAT         Default: 1.0
  --max-out-len N             Used by both stages; default: 32768
  --extra-body-json JSON      Sampling parameters sent as extra_body
  --retry N                   API retry count; default: 1
  --request-timeout N         Per-request timeout; default: 7200
  --time-budget-seconds N     Hard total budget; default: 7200
  --short-budget-seconds N    Maximum short-stage time; default: 3000

Short-stage options:
  --short-benchmarks LIST     Ordered comma list; default:
                              aime_2025,ifbench,ceval
  --short-max-seq-len N       Default: 65536
  --short-batch-size N        Default: 1024
  --short-workers N           Default: 512
  --short-qps N               Default: 128
  --ceval-dataset-workers N   Parallel C-Eval subject partitions; default: 32
                              Other short benchmarks remain at 1.
  --skip-short                Run/resume only the long stage

Long-stage options:
  --long-benchmarks LIST      Ordered comma list; default: longbench_v2
                              AA-LCR may be added explicitly as aa_lcr.
  --long-max-seq-len N        Default: 262144
  --long-batch-size N         Default: 32
  --long-workers N            Default: 32 total client requests
  --long-qps N                Default: 8
  --long-samples N|all        Default: all. A number is diagnostic sampling,
                              not an official full benchmark result.
  --skip-long                 Run/resume only the short stage

Monitoring and inspection:
  --metrics-url URL           Default: BASE_URL without /v1 plus /metrics
  --monitor-interval N        Metrics sampling interval; default: 60 seconds
  --no-metrics-monitor        Do not record vLLM metrics
  --dry-run                   Print child commands without inference
  -h, --help                  Show this help

Notes:
  * Every benchmark runs infer -> eval -> numeric-summary verification before
    the next benchmark starts.
  * INCLUDE is not part of this profile. Its official /v1/completions logprob
    protocol ignores chat sampling and generation length.
  * AA-LCR scoring additionally requires the official judge environment used
    by run_benchmarks.sh.
  * The 7200-second limit guarantees bounded runtime, not completion. Full
    LongBench v2 with 32768-token thinking needs measured serving throughput;
    the launcher prints the relevant worst-case output-token rate.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local count="$2"
  (( count >= 2 )) || die "$option requires a value"
}

while (( $# > 0 )); do
  case "$1" in
    --model) require_value "$1" "$#"; MODEL_NAME="$2"; shift 2 ;;
    --base-url) require_value "$1" "$#"; BASE_URL="$2"; shift 2 ;;
    --api-key) require_value "$1" "$#"; API_KEY="$2"; shift 2 ;;
    --tokenizer-path) require_value "$1" "$#"; TOKENIZER_PATH="$2"; shift 2 ;;
    --python) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
    --output-root) require_value "$1" "$#"; OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id|--reuse) require_value "$1" "$#"; RUN_ID="$2"; shift 2 ;;
    --temperature) require_value "$1" "$#"; TEMPERATURE="$2"; shift 2 ;;
    --max-out-len) require_value "$1" "$#"; MAX_OUT_LEN="$2"; shift 2 ;;
    --extra-body-json) require_value "$1" "$#"; EXTRA_BODY_JSON="$2"; shift 2 ;;
    --retry) require_value "$1" "$#"; RETRY="$2"; shift 2 ;;
    --request-timeout) require_value "$1" "$#"; REQUEST_TIMEOUT="$2"; shift 2 ;;
    --time-budget-seconds) require_value "$1" "$#"; TIME_BUDGET_SECONDS="$2"; shift 2 ;;
    --short-budget-seconds) require_value "$1" "$#"; SHORT_BUDGET_SECONDS="$2"; shift 2 ;;
    --short-benchmarks) require_value "$1" "$#"; SHORT_BENCHMARKS="$2"; shift 2 ;;
    --short-max-seq-len) require_value "$1" "$#"; SHORT_MAX_SEQ_LEN="$2"; shift 2 ;;
    --short-batch-size) require_value "$1" "$#"; SHORT_BATCH_SIZE="$2"; shift 2 ;;
    --short-workers) require_value "$1" "$#"; SHORT_WORKERS="$2"; shift 2 ;;
    --short-qps) require_value "$1" "$#"; SHORT_QPS="$2"; shift 2 ;;
    --ceval-dataset-workers) require_value "$1" "$#"; CEVAL_DATASET_WORKERS="$2"; shift 2 ;;
    --long-benchmarks) require_value "$1" "$#"; LONG_BENCHMARKS="$2"; shift 2 ;;
    --long-max-seq-len) require_value "$1" "$#"; LONG_MAX_SEQ_LEN="$2"; shift 2 ;;
    --long-batch-size) require_value "$1" "$#"; LONG_BATCH_SIZE="$2"; shift 2 ;;
    --long-workers) require_value "$1" "$#"; LONG_WORKERS="$2"; shift 2 ;;
    --long-qps) require_value "$1" "$#"; LONG_QPS="$2"; shift 2 ;;
    --long-samples) require_value "$1" "$#"; LONG_SAMPLES="$2"; shift 2 ;;
    --metrics-url) require_value "$1" "$#"; METRICS_URL="$2"; shift 2 ;;
    --monitor-interval) require_value "$1" "$#"; MONITOR_INTERVAL_SECONDS="$2"; shift 2 ;;
    --no-metrics-monitor) MONITOR_METRICS=0; shift ;;
    --skip-short) SKIP_SHORT=1; shift ;;
    --skip-long) SKIP_LONG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$MODEL_NAME" ]] || die '--model is required'
[[ -n "$BASE_URL" ]] || die '--base-url is required'
[[ -n "$OUTPUT_ROOT" ]] || die '--output-root cannot be empty'
[[ -n "$RUN_ID" ]] || die '--run-id cannot be empty'
[[ "$RUN_ID" != */* ]] || die '--run-id must not contain /'
if [[ -z "$TOKENIZER_PATH" ]]; then
  TOKENIZER_PATH="$MODEL_NAME"
fi
(( ! SKIP_SHORT || ! SKIP_LONG )) || die '--skip-short and --skip-long cannot be used together'

for variable in MAX_OUT_LEN RETRY REQUEST_TIMEOUT TIME_BUDGET_SECONDS \
    SHORT_BUDGET_SECONDS SHORT_MAX_SEQ_LEN SHORT_BATCH_SIZE SHORT_WORKERS \
    SHORT_QPS CEVAL_DATASET_WORKERS LONG_MAX_SEQ_LEN LONG_BATCH_SIZE LONG_WORKERS LONG_QPS \
    SHORT_STREAM_IDLE_TIMEOUT LONG_STREAM_IDLE_TIMEOUT \
    MONITOR_INTERVAL_SECONDS TIMEOUT_KILL_AFTER_SECONDS; do
  value="${!variable}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer, got: $value"
done
[[ "$LONG_SAMPLES" == all || "$LONG_SAMPLES" =~ ^[1-9][0-9]*$ ]] || \
  die '--long-samples must be a positive integer or all'
(( SHORT_BUDGET_SECONDS <= TIME_BUDGET_SECONDS )) || \
  die '--short-budget-seconds cannot exceed --time-budget-seconds'
if (( ! SKIP_LONG && SHORT_BUDGET_SECONDS >= TIME_BUDGET_SECONDS )); then
  die '--short-budget-seconds must leave time for the long stage'
fi

split_benchmarks() {
  local value="$1"
  local -n destination="$2"
  local item
  local -a parsed=()
  IFS=',' read -r -a parsed <<< "$value"
  (( ${#parsed[@]} > 0 )) || die 'benchmark list cannot be empty'
  for item in "${parsed[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || die 'benchmark list contains an empty name'
    destination+=("$item")
  done
}

SHORT_ITEMS=()
LONG_ITEMS=()
split_benchmarks "$SHORT_BENCHMARKS" SHORT_ITEMS
split_benchmarks "$LONG_BENCHMARKS" LONG_ITEMS

if [[ -z "$METRICS_URL" ]]; then
  normalized_base_url="${BASE_URL%/}"
  if [[ "$normalized_base_url" == */v1 ]]; then
    METRICS_URL="${normalized_base_url%/v1}/metrics"
  else
    METRICS_URL="$normalized_base_url/metrics"
  fi
fi

STATE_DIR="$OUTPUT_ROOT/_two_hour_runs/$RUN_ID"
METRICS_LOG="$STATE_DIR/vllm_metrics.log"
STATUS_LOG="$STATE_DIR/status.log"
mkdir -p "$STATE_DIR"

print_command_redacted() {
  local redact_next=0
  local argument
  for argument in "$@"; do
    if (( redact_next )); then
      printf ' %s' '<redacted>'
      redact_next=0
    else
      printf ' %q' "$argument"
      [[ "$argument" == '--api-key' ]] && redact_next=1
    fi
  done
  printf '\n'
}

monitor_metrics() {
  while :; do
    {
      printf '\n# %s\n' "$(date --iso-8601=seconds)"
      curl -fsS --max-time 10 "$METRICS_URL" 2>/dev/null | \
        awk '/^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions|generation_tokens|prompt_tokens)/'
    } >> "$METRICS_LOG" || true
    sleep "$MONITOR_INTERVAL_SECONDS"
  done
}

MONITOR_PID=""
stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap stop_monitor EXIT

if (( MONITOR_METRICS && ! DRY_RUN )); then
  monitor_metrics &
  MONITOR_PID=$!
fi

START_EPOCH="$(date +%s)"
OVERALL_DEADLINE=$((START_EPOCH + TIME_BUDGET_SECONDS))
SHORT_DEADLINE=$((START_EPOCH + SHORT_BUDGET_SECONDS))

longbench_samples=503
if [[ "$LONG_SAMPLES" != all && "$LONG_SAMPLES" -lt "$longbench_samples" ]]; then
  longbench_samples="$LONG_SAMPLES"
fi
if (( SKIP_LONG )); then
  long_window="$TIME_BUDGET_SECONDS"
elif (( SKIP_SHORT )); then
  long_window="$TIME_BUDGET_SECONDS"
else
  long_window=$((TIME_BUDGET_SECONDS - SHORT_BUDGET_SECONDS))
fi
longbench_token_cap=$((longbench_samples * MAX_OUT_LEN))
longbench_required_tps=$(((longbench_token_cap + long_window - 1) / long_window))

echo "Run id: $RUN_ID"
echo "Output root: $OUTPUT_ROOT"
echo "Hard wall-clock budget: ${TIME_BUDGET_SECONDS}s; short-stage cap: ${SHORT_BUDGET_SECONDS}s"
echo "Short stage: ${SHORT_ITEMS[*]} (workers=$SHORT_WORKERS, qps=$SHORT_QPS, context=$SHORT_MAX_SEQ_LEN, output=$MAX_OUT_LEN)"
echo "Long stage: ${LONG_ITEMS[*]} (workers=$LONG_WORKERS, qps=$LONG_QPS, context=$LONG_MAX_SEQ_LEN, output=$MAX_OUT_LEN, samples=$LONG_SAMPLES)"
echo "LongBench worst-case decode gate: about ${longbench_required_tps} output token/s during the reserved ${long_window}s long window"
echo "This is a max-token upper bound; real completion depends on emitted tokens, prefill throughput and KV-cache stability."
if (( MONITOR_METRICS && ! DRY_RUN )); then
  echo "vLLM metrics: $METRICS_URL -> $METRICS_LOG"
fi

printf '%s run_started budget=%s short_budget=%s\n' \
  "$(date --iso-8601=seconds)" "$TIME_BUDGET_SECONDS" \
  "$SHORT_BUDGET_SECONDS" >> "$STATUS_LOG"

build_command() {
  local phase="$1"
  local benchmark="$2"
  TASK_COMMAND=(
    bash "$SCRIPT_DIR/run_benchmarks.sh"
    --model "$MODEL_NAME"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --tokenizer-path "$TOKENIZER_PATH"
    --python "$PYTHON_BIN"
    --benchmark "$benchmark"
    --output-root "$OUTPUT_ROOT/$phase"
    --run-id "$RUN_ID"
    --temperature "$TEMPERATURE"
    --max-out-len "$MAX_OUT_LEN"
    --retry "$RETRY"
    --timeout "$REQUEST_TIMEOUT"
    --extra-body-json "$EXTRA_BODY_JSON"
    --stream-responses
  )

  if [[ "$phase" == short ]]; then
    TASK_COMMAND+=(
      --max-seq-len "$SHORT_MAX_SEQ_LEN"
      --batch-size "$SHORT_BATCH_SIZE"
      --max-workers "$SHORT_WORKERS"
      --query-per-second "$SHORT_QPS"
      --stream-idle-timeout "$SHORT_STREAM_IDLE_TIMEOUT"
      --samples all
    )
    # Only C-Eval needs partition-level parallelism in the default profile.
    # Its individual subjects are too small to use SHORT_WORKERS on their own.
    if [[ "$benchmark" == ceval || "$benchmark" == c_eval || \
        "$benchmark" == ceval_evalscope ]]; then
      TASK_COMMAND+=(--dataset-workers "$CEVAL_DATASET_WORKERS")
    else
      TASK_COMMAND+=(--dataset-workers 1)
    fi
  else
    TASK_COMMAND+=(
      --max-seq-len "$LONG_MAX_SEQ_LEN"
      --batch-size "$LONG_BATCH_SIZE"
      --max-workers "$LONG_WORKERS"
      --query-per-second "$LONG_QPS"
      --stream-idle-timeout "$LONG_STREAM_IDLE_TIMEOUT"
      --samples "$LONG_SAMPLES"
      --dataset-workers 1
    )
  fi
  if (( DRY_RUN )); then
    TASK_COMMAND+=(--dry-run)
  fi
}

run_with_deadline() {
  local label="$1"
  local deadline="$2"
  shift 2
  local now remaining status
  now="$(date +%s)"
  remaining=$((deadline - now))
  if (( remaining <= 0 )); then
    echo "TIME BUDGET: no time remains for $label" >&2
    return 124
  fi

  printf '%s benchmark_started label=%q remaining=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$remaining" >> "$STATUS_LOG"
  echo
  echo "=== $label (deadline in ${remaining}s) ==="
  if (( DRY_RUN )); then
    printf 'DRY-RUN child:'
    print_command_redacted "$@"
    "$@"
    return $?
  fi

  if timeout --signal=INT --kill-after="${TIMEOUT_KILL_AFTER_SECONDS}s" \
      "${remaining}s" "$@"; then
    status=0
  else
    status=$?
  fi
  printf '%s benchmark_finished label=%q status=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$status" >> "$STATUS_LOG"
  return "$status"
}

had_failure=0
had_timeout=0

if (( ! SKIP_SHORT )); then
  for benchmark in "${SHORT_ITEMS[@]}"; do
    build_command short "$benchmark"
    if run_with_deadline "short/$benchmark" "$SHORT_DEADLINE" \
        "${TASK_COMMAND[@]}"; then
      :
    else
      status=$?
      if (( status == 124 )); then
        had_timeout=1
        echo "Short-stage cap reached while running $benchmark; proceeding to the long stage." >&2
        break
      fi
      had_failure=1
      echo "Short benchmark $benchmark failed with status $status; continuing." >&2
    fi
  done
fi

if (( ! SKIP_LONG )); then
  for benchmark in "${LONG_ITEMS[@]}"; do
    build_command long "$benchmark"
    if run_with_deadline "long/$benchmark" "$OVERALL_DEADLINE" \
        "${TASK_COMMAND[@]}"; then
      :
    else
      status=$?
      if (( status == 124 )); then
        had_timeout=1
        echo "Overall two-hour budget reached while running $benchmark." >&2
        break
      fi
      had_failure=1
      echo "Long benchmark $benchmark failed with status $status; continuing." >&2
    fi
  done
fi

stop_monitor
elapsed=$(( $(date +%s) - START_EPOCH ))
printf '%s run_finished elapsed=%s failure=%s timeout=%s\n' \
  "$(date --iso-8601=seconds)" "$elapsed" "$had_failure" \
  "$had_timeout" >> "$STATUS_LOG"

echo
echo "Elapsed: ${elapsed}s"
echo "Status log: $STATUS_LOG"
echo "Resume command: bash script/run_posttrain_2h.sh --model $(printf %q "$MODEL_NAME") --base-url $(printf %q "$BASE_URL") --run-id $(printf %q "$RUN_ID")"
if (( DRY_RUN )); then
  echo "Dry run complete: no inference or scoring was performed."
  exit 0
fi
if (( had_timeout )); then
  echo "The wall-clock budget expired. Existing predictions were kept for --run-id $RUN_ID." >&2
  exit 124
fi
if (( had_failure )); then
  echo "The plan finished with at least one failed benchmark; inspect $STATUS_LOG." >&2
  exit 1
fi
echo "Two-stage plan completed and every selected benchmark produced a numeric score."
