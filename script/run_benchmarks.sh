#!/usr/bin/env bash
set -euo pipefail

# Model-agnostic, sequential launcher for the benchmark validation set.
# Every child invocation runs exactly one benchmark in OpenCompass "all" mode
# (infer -> eval -> summary). A numeric summary is required before the next
# benchmark starts, so inference for every task is scored immediately instead
# of deferring all evaluation until the end of the suite.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHMARKS=(
  mmlu_pro
  ceval
  supergpqa
  ifeval
  mmmlu_downsampling
  gpqa_diamond
  ifbench
  aime_2024
  aime_2025
  aime_2026
  scicode
  livecodebench
  humaneval
  mmlu_redux_downsampling
  mmlu_prox_downsampling
  global_piqa_downsampling
  hmmt_feb_2025
  hmmt_nov_2025
  include_downsampling
  hmmt_feb_2026
  longbench_v2
)

# Canonical suite names map directly to the one shared execution engine. The
# legacy run_<benchmark>.sh files remain as compatibility entry points, but
# this launcher deliberately does not hop through those one-line wrappers.
declare -A CORE_TARGETS=(
  [mmlu_pro]=mmlu_pro
  [ceval]=ceval_evalscope
  [supergpqa]=supergpqa
  [ifeval]=ifeval
  [mmmlu_downsampling]=mmmlu_downsampling
  [gpqa_diamond]=gpqa_diamond
  [ifbench]=ifbench
  [longbench_v2]=longbenchv2
  [aime_2024]=aime2024
  [aime_2025]=aime2025
  [aime_2026]=aime2026
  [hmmt_feb_2026]=hmmt2026
  [scicode]=scicode
  [livecodebench]=livecodebench
  [humaneval]=humaneval
  [mmlu_prox_downsampling]=mmlu_prox_downsampling
  [global_piqa_downsampling]=global_piqa_downsampling
  [mmlu_redux_downsampling]=mmlu_redux_downsampling
  [hmmt_feb_2025]=hmmt_feb_2025
  [hmmt_nov_2025]=hmmt_nov_2025
  [include_downsampling]=include_downsampling
)

SELECTION_ARGS=()
COMMON_ARGS=()
MODEL_NAME=""
BASE_URL=""
TOKENIZER_PATH=""
OUTPUT_ROOT="outputs/benchmark_suite"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
DRY_RUN=0
KEEP_GOING=0
LIST_ONLY=0
VERIFY_PYTHON="${PYTHON_BIN:-${REPO_ROOT}/../opencompass-venv/bin/python}"
AGENT_EVAL_UPLOAD=0
AGENT_EVAL_PLATFORM_URL="${AGENT_EVAL_PLATFORM_URL:-https://47.88.93.207}"
AGENT_EVAL_EVIDENCE_URL="${AGENT_EVAL_EVIDENCE_URL:-}"
AGENT_EVAL_EVIDENCE_ROOT="${AGENT_EVAL_EVIDENCE_ROOT:-}"
AGENT_EVAL_SYSTEM="${AGENT_EVAL_SYSTEM:-}"
AGENT_EVAL_CHECKPOINT_ID="${AGENT_EVAL_CHECKPOINT_ID:-}"
AGENT_EVAL_SCOPE="${AGENT_EVAL_SCOPE:-posttrained}"
AGENT_EVAL_PROVENANCE="${AGENT_EVAL_PROVENANCE:-internal_private}"
AGENT_EVAL_PUBLISHER="${AGENT_EVAL_PUBLISHER:-OpenCompass Evaluation Team}"
AGENT_EVAL_MAPPING_FILE="${AGENT_EVAL_MAPPING_FILE:-}"
AGENT_EVAL_TOKEN_ENV="${AGENT_EVAL_TOKEN_ENV:-AGENT_EVAL_SUBMISSION_PAT}"
AGENT_EVAL_MODE="${AGENT_EVAL_MODE:-ingest}"
AGENT_EVAL_INGEST_TOKEN_ENV="${AGENT_EVAL_INGEST_TOKEN_ENV:-INGEST_ADMIN_TOKEN}"
AGENT_EVAL_DRY_RUN=0
VLLM_REPLICA_COUNT="${OPENCOMPASS_VLLM_REPLICA_COUNT:-1}"
SHORT_CONCURRENCY_PER_REPLICA="${OPENCOMPASS_SHORT_CONCURRENCY_PER_REPLICA:-256}"
LONG_OUTPUT_CONCURRENCY_PER_REPLICA="${OPENCOMPASS_LONG_OUTPUT_CONCURRENCY_PER_REPLICA:-128}"
LONG_CONTEXT_CONCURRENCY_PER_REPLICA="${OPENCOMPASS_LONG_CONTEXT_CONCURRENCY_PER_REPLICA:-32}"
MAX_DATASET_WORKERS="${OPENCOMPASS_MAX_DATASET_WORKERS:-14}"
BATCH_SIZE_OVERRIDE="${OPENCOMPASS_BATCH_SIZE_OVERRIDE:-}"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_benchmarks.sh --model NAME --base-url URL [options]

Selection:
  --benchmark NAME[,NAME...]  Benchmark selection. Repeatable; default: all
  --benchmarks NAME[,NAME...] Alias of --benchmark
  --list-benchmarks           Print canonical names and exit

Examples:
  # Run the complete suite, scoring each benchmark immediately.
  bash script/run_benchmarks.sh \
    --model deployed-model-name \
    --base-url http://127.0.0.1:8000/v1

  # Run only three benchmarks, in canonical suite order.
  bash script/run_benchmarks.sh \
    --model deployed-model-name \
    --base-url http://127.0.0.1:8000/v1 \
    --benchmark mmlu_pro,ceval,ifeval

  # Reuse one run id under every per-benchmark work directory.
  bash script/run_benchmarks.sh \
    --model deployed-model-name \
    --base-url http://127.0.0.1:8000/v1 \
    --benchmark longbench_v2 --reuse 20260727_120000

Execution policy:
  Benchmarks are deliberately sequential. Every selected benchmark completes
  infer -> eval -> summary and must produce a numeric summary CSV before the
  next benchmark begins. There is no final deferred-evaluation phase.

Execution engines:
  Every benchmark calls run_posttrain_objective_benchmark.sh directly. That
  shared core selects chat generation or INCLUDE prompt-logprob completions.
  This launcher does not depend on the legacy one-line run_<benchmark>.sh
  wrappers or the multilingual smoke-suite launcher.

Canonical benchmark names:
  mmlu_pro, ceval, supergpqa, ifeval, mmmlu_downsampling, gpqa_diamond,
  ifbench, longbench_v2, aime_2024, aime_2025, aime_2026,
  hmmt_feb_2026, scicode, livecodebench, humaneval,
  mmlu_redux_downsampling,
  mmlu_prox_downsampling, global_piqa_downsampling, hmmt_feb_2025,
  hmmt_nov_2025, include_downsampling

Runtime options:
  --output-root DIR           Parent of per-benchmark work dirs
                              (default: outputs/benchmark_suite)
  --run-id ID                Run id used below every work dir
  --reuse ID                 Alias of --run-id; resumes missing work
  --keep-going               Continue after failures and report them at end
  --dry-run                  Print the exact execution plan without running
  --base-url URL             Required OpenAI-compatible base URL
  --model NAME               Required served model name
  --api-key KEY              Forward without printing it in the plan
  --python PATH              Python executable used by child launchers
  --tokenizer-path PATH      Tokenizer used for context accounting;
                              defaults to --model when omitted
  --max-seq-len N            Global override; otherwise retain task defaults
  --max-out-len N            Global override; otherwise retain task defaults
  --omit-max-out-len       Do not send an output-token limit
  --temperature FLOAT        Global override; otherwise retain task defaults
  --query-per-second N       Global override
  --batch-size N             Global override
  --max-workers N            Global override
  --dataset-workers N        Concurrent OpenCompass dataset partitions.
                             Total API concurrency can approach this value
                             times --max-workers; use with care.
  --vllm-replicas N         Number of vLLM replicas assigned to this runner.
                             Default: OPENCOMPASS_VLLM_REPLICA_COUNT or 1.
  --short-concurrency-per-replica N
                             Short benchmark target per replica; default: 256.
  --long-output-concurrency-per-replica N
                             HMMT-style long-output target; default: 128.
  --long-context-concurrency-per-replica N
                             LongBenchV2 target; default: 32.
  --retry N                  Global override
  --timeout SECONDS          Global override
  --stream-idle-timeout N   Global streamed-response idle timeout override
  --samples N|all            Per-dataset sample limit; default: all
  --extra-body-json JSON     Global sampling extra_body override
  --openai-extra-kwargs-json JSON
                              Global OpenAI request kwargs override
  --stream-responses         Stream chat responses
  --no-stream-responses      Disable streaming
  --agent-eval-upload        After each numeric score, send it to Agent Eval
  --agent-eval-mode MODE     ingest (default, direct canonical write without
                             Feishu) or review (pending human review)
  --agent-eval-platform-url URL
                             Default: AGENT_EVAL_PLATFORM_URL or
                             https://47.88.93.207
  --agent-eval-evidence-url URL
                             Required with upload. May contain {benchmark},
                             {run_id}, {summary_file}, {summary_sha256}
  --agent-eval-evidence-root DIR
                             Publish scored summary files below this directory
  --agent-eval-system NAME   Exact platform system ID/name; defaults to --model
  --agent-eval-checkpoint-id ID
                             Canonical checkpoint associated with this run
  --agent-eval-scope SCOPE   pretrained, posttrained, or agent_system
  --agent-eval-provenance P  internal_public or internal_private
  --agent-eval-publisher S   Evidence publisher label
  --agent-eval-mapping-file FILE
                             Per-benchmark catalog version/alias overrides
  --agent-eval-token-env VAR Read the PAT from VAR; it is never put in argv
                             This PAT performs canonical catalog lookup
  --agent-eval-ingest-token-env VAR
                             Direct-ingest service token environment variable;
                             default: INGEST_ADMIN_TOKEN
  --agent-eval-dry-run       Validate the upload contract without writing data
  -h, --help                 Show this help

Notes:
  * ceval selects the EvalScope-compatible C-Eval prompt implementation.
  * gpqa_diamond is the GPQA target used by the validation suite.
  * Concurrency is computed per benchmark from the number of assigned vLLM
    replicas. Dataset workers divide that target instead of multiplying it.
  * hmmt_feb_2026 uses 131072 context, 81920 output tokens and the long-output
    concurrency profile (128 active requests per replica by default).
  * longbench_v2 uses 262144 context, 32768 output tokens and the long-context
    profile (32 active requests per replica by default).
  * Agent Eval upload always performs a server dry-run first. The default
    ingest mode writes canonical run/result rows directly and never creates a
    Feishu review. Use --agent-eval-mode review for the legacy review path.
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

canonicalize_benchmark() {
  local name="${1,,}"
  name="${name//[[:space:]]/}"
  name="${name//-/_}"
  name="${name//./_}"
  case "$name" in
    all) echo all ;;
    mmlu_pro|mmlupro) echo mmlu_pro ;;
    ceval|c_eval|ceval_evalscope) echo ceval ;;
    supergpqa|super_gpqa) echo supergpqa ;;
    ifeval|if_eval) echo ifeval ;;
    mmmlu_downsampling|mmmlu_sampled) echo mmmlu_downsampling ;;
    gpqa|gpqa_diamond) echo gpqa_diamond ;;
    ifbench|if_bench) echo ifbench ;;
    longbench|longbenchv2|longbench_v2) echo longbench_v2 ;;
    aime24|aime2024|aime_2024) echo aime_2024 ;;
    aime25|aime2025|aime_2025) echo aime_2025 ;;
    aime26|aime2026|aime_2026) echo aime_2026 ;;
    hmmt2026|hmmt_feb2026|hmmt_feb_2026) echo hmmt_feb_2026 ;;
    scicode|sci_code) echo scicode ;;
    livecodebench|live_code_bench|livecodebench_v6) echo livecodebench ;;
    humaneval|human_eval|openai_humaneval) echo humaneval ;;
    mmlu_redux_downsampling|mmluredux_downsampling) echo mmlu_redux_downsampling ;;
    mmlu_prox_downsampling|mmluprox_downsampling) echo mmlu_prox_downsampling ;;
    global_piqa_downsampling|globalpiqa_downsampling) echo global_piqa_downsampling ;;
    hmmt_feb2025|hmmt_feb_2025) echo hmmt_feb_2025 ;;
    hmmt_nov2025|hmmt_nov_2025) echo hmmt_nov_2025 ;;
    include_downsampling|include_base_44_downsampling) echo include_downsampling ;;
    *) return 1 ;;
  esac
}

append_selections() {
  local value="$1"
  local part canonical
  local -a parts=()
  IFS=',' read -r -a parts <<< "$value"
  for part in "${parts[@]}"; do
    [[ -n "${part//[[:space:]]/}" ]] || die 'empty benchmark name'
    canonical="$(canonicalize_benchmark "$part")" || \
      die "unknown benchmark: $part (use --list-benchmarks)"
    SELECTION_ARGS+=("$canonical")
  done
}

while (( $# > 0 )); do
  case "$1" in
    --benchmark|--benchmarks)
      require_value "$1" "$#"
      append_selections "$2"
      shift 2
      ;;
    --list-benchmarks)
      LIST_ONLY=1
      shift
      ;;
    --output-root)
      require_value "$1" "$#"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-id|--reuse)
      require_value "$1" "$#"
      RUN_ID="$2"
      shift 2
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python)
      require_value "$1" "$#"
      VERIFY_PYTHON="$2"
      COMMON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --model)
      require_value "$1" "$#"
      MODEL_NAME="$2"
      shift 2
      ;;
    --base-url)
      require_value "$1" "$#"
      BASE_URL="$2"
      shift 2
      ;;
    --tokenizer-path)
      require_value "$1" "$#"
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --api-key|--max-seq-len|--max-out-len|--temperature|--query-per-second|--batch-size|--max-workers|--dataset-workers|--retry|--timeout|--stream-idle-timeout|--samples|--extra-body-json|--openai-extra-kwargs-json)
      require_value "$1" "$#"
      COMMON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --omit-max-out-len)
      COMMON_ARGS+=("$1")
      shift
      ;;
    --stream-responses|--no-stream-responses)
      COMMON_ARGS+=("$1")
      shift
      ;;
    --vllm-replicas)
      require_value "$1" "$#"
      VLLM_REPLICA_COUNT="$2"
      shift 2
      ;;
    --short-concurrency-per-replica)
      require_value "$1" "$#"
      SHORT_CONCURRENCY_PER_REPLICA="$2"
      shift 2
      ;;
    --long-output-concurrency-per-replica)
      require_value "$1" "$#"
      LONG_OUTPUT_CONCURRENCY_PER_REPLICA="$2"
      shift 2
      ;;
    --long-context-concurrency-per-replica)
      require_value "$1" "$#"
      LONG_CONTEXT_CONCURRENCY_PER_REPLICA="$2"
      shift 2
      ;;
    --agent-eval-upload)
      AGENT_EVAL_UPLOAD=1
      shift
      ;;
    --agent-eval-mode)
      require_value "$1" "$#"
      AGENT_EVAL_MODE="$2"
      shift 2
      ;;
    --agent-eval-platform-url)
      require_value "$1" "$#"
      AGENT_EVAL_PLATFORM_URL="$2"
      shift 2
      ;;
    --agent-eval-evidence-url)
      require_value "$1" "$#"
      AGENT_EVAL_EVIDENCE_URL="$2"
      shift 2
      ;;
    --agent-eval-evidence-root)
      require_value "$1" "$#"
      AGENT_EVAL_EVIDENCE_ROOT="$2"
      shift 2
      ;;
    --agent-eval-system)
      require_value "$1" "$#"
      AGENT_EVAL_SYSTEM="$2"
      shift 2
      ;;
    --agent-eval-checkpoint-id)
      require_value "$1" "$#"
      AGENT_EVAL_CHECKPOINT_ID="$2"
      shift 2
      ;;
    --agent-eval-scope)
      require_value "$1" "$#"
      AGENT_EVAL_SCOPE="$2"
      shift 2
      ;;
    --agent-eval-provenance)
      require_value "$1" "$#"
      AGENT_EVAL_PROVENANCE="$2"
      shift 2
      ;;
    --agent-eval-publisher)
      require_value "$1" "$#"
      AGENT_EVAL_PUBLISHER="$2"
      shift 2
      ;;
    --agent-eval-mapping-file)
      require_value "$1" "$#"
      AGENT_EVAL_MAPPING_FILE="$2"
      shift 2
      ;;
    --agent-eval-token-env)
      require_value "$1" "$#"
      AGENT_EVAL_TOKEN_ENV="$2"
      shift 2
      ;;
    --agent-eval-ingest-token-env)
      require_value "$1" "$#"
      AGENT_EVAL_INGEST_TOKEN_ENV="$2"
      shift 2
      ;;
    --agent-eval-dry-run)
      AGENT_EVAL_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "$LIST_ONLY" == 1 ]]; then
  printf '%s\n' "${ALL_BENCHMARKS[@]}"
  exit 0
fi

[[ -n "$MODEL_NAME" ]] || die '--model is required'
[[ -n "$BASE_URL" ]] || die '--base-url is required'
for concurrency_value in \
    VLLM_REPLICA_COUNT \
    SHORT_CONCURRENCY_PER_REPLICA \
    LONG_OUTPUT_CONCURRENCY_PER_REPLICA \
    LONG_CONTEXT_CONCURRENCY_PER_REPLICA; do
  value="${!concurrency_value}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || \
    die "$concurrency_value must be a positive integer, got: $value"
done
if [[ -z "$TOKENIZER_PATH" ]]; then
  TOKENIZER_PATH="$MODEL_NAME"
fi

if (( AGENT_EVAL_UPLOAD )); then
  [[ -n "$AGENT_EVAL_PLATFORM_URL" ]] || die \
    '--agent-eval-platform-url cannot be empty'
  [[ -n "$AGENT_EVAL_EVIDENCE_URL" ]] || die \
    '--agent-eval-evidence-url (or AGENT_EVAL_EVIDENCE_URL) is required for provenance'
  if [[ -n "$AGENT_EVAL_EVIDENCE_ROOT" ]]; then
    [[ -d "$AGENT_EVAL_EVIDENCE_ROOT" && -w "$AGENT_EVAL_EVIDENCE_ROOT" ]] || die \
      '--agent-eval-evidence-root must be an existing writable directory'
  fi
  [[ "$AGENT_EVAL_TOKEN_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die \
    '--agent-eval-token-env must be a shell environment variable name'
  [[ "$AGENT_EVAL_INGEST_TOKEN_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die \
    '--agent-eval-ingest-token-env must be a shell environment variable name'
  case "$AGENT_EVAL_MODE" in
    ingest|review) ;;
    *) die '--agent-eval-mode must be ingest or review' ;;
  esac
  case "$AGENT_EVAL_SCOPE" in
    pretrained|posttrained|agent_system) ;;
    *) die '--agent-eval-scope must be pretrained, posttrained, or agent_system' ;;
  esac
  case "$AGENT_EVAL_PROVENANCE" in
    internal_public|internal_private) ;;
    *) die '--agent-eval-provenance must be internal_public or internal_private' ;;
  esac
  if (( ! DRY_RUN )) && [[ -z "${!AGENT_EVAL_TOKEN_ENV:-}" ]]; then
    die "$AGENT_EVAL_TOKEN_ENV must be exported when --agent-eval-upload is used"
  fi
  if (( ! DRY_RUN )) && [[ "$AGENT_EVAL_MODE" == ingest ]] \
      && [[ -z "${!AGENT_EVAL_INGEST_TOKEN_ENV:-}" ]]; then
    die "$AGENT_EVAL_INGEST_TOKEN_ENV must be exported for direct Agent Eval ingest"
  fi
fi

COMMON_ARGS=(
  --base-url "$BASE_URL"
  --model "$MODEL_NAME"
  --tokenizer-path "$TOKENIZER_PATH"
  "${COMMON_ARGS[@]}"
)

[[ -n "$OUTPUT_ROOT" ]] || die '--output-root cannot be empty'
[[ -n "$RUN_ID" ]] || die '--run-id cannot be empty'
[[ "$RUN_ID" != */* ]] || die '--run-id must not contain /'

if (( ${#SELECTION_ARGS[@]} == 0 )); then
  SELECTION_ARGS=(all)
fi

declare -A requested=()
for benchmark in "${SELECTION_ARGS[@]}"; do
  if [[ "$benchmark" == all ]]; then
    for item in "${ALL_BENCHMARKS[@]}"; do
      requested["$item"]=1
    done
  else
    requested["$benchmark"]=1
  fi
done

# Emit selected tasks in one stable order even if names were repeated or
# supplied in a different order. Regular 64K tasks run first, HMMT runs after
# them with its 128K profile, and LongBench v2 always runs last with its own
# audited defaults.
SELECTED=()
for benchmark in "${ALL_BENCHMARKS[@]}"; do
  if [[ -n "${requested[$benchmark]:-}" ]]; then
    SELECTED+=("$benchmark")
  fi
done

(( ${#SELECTED[@]} > 0 )) || die 'no benchmark selected'

print_command_redacted() {
  local redact_next=0
  local arg
  for arg in "$@"; do
    if (( redact_next )); then
      printf ' %s' '<redacted>'
      redact_next=0
    else
      printf ' %q' "$arg"
      [[ "$arg" == '--api-key' ]] && redact_next=1
    fi
  done
  printf '\n'
}

build_command() {
  local benchmark="$1"
  local work_dir="$2"
  local core_target
  local dataset_workers=1
  local concurrency_per_replica="$SHORT_CONCURRENCY_PER_REPLICA"
  local concurrency_profile=short
  local total_concurrency
  local workers_per_dataset
  local request_batch_size
  TASK_ARGS=()

  case "$benchmark" in
    ceval)
      # C-Eval has 52 subject partitions. Expose all of them so the runner can
      # keep every assigned replica busy instead of draining a 32-task tail.
      dataset_workers=52
      ;;
    mmmlu_downsampling)
      # The sampled suite has 14 language partitions x 100 rows. Run every
      # language concurrently; each partition still receives enough API
      # workers to saturate its available rows.
      dataset_workers=14
      ;;
    mmlu_prox_downsampling)
      # 406 tiny language/subject partitions average 3.4 sampled rows each.
      # A 192-partition window exposes roughly 650 rows at once without
      # spawning all 406 OpenCompass Python workers on the 16-core host.
      dataset_workers=192
      ;;
    include_downsampling)
      # 44 partitions average 31.8 sampled rows each.
      dataset_workers=16
      ;;
    mmlu_redux_downsampling|global_piqa_downsampling)
      # Each is one 1400-row partition; API workers provide the concurrency.
      dataset_workers=1
      ;;
    hmmt_feb_2025|hmmt_nov_2025|hmmt_feb_2026)
      concurrency_profile=long-output
      concurrency_per_replica="$LONG_OUTPUT_CONCURRENCY_PER_REPLICA"
      if [[ "$benchmark" == hmmt_feb_2026 ]]; then
        TASK_ARGS=(
          --max-seq-len 131072
          --max-out-len 81920
          --timeout 7200
          --stream-idle-timeout 1200
        )
      fi
      ;;
    longbench_v2)
      concurrency_profile=long-context
      concurrency_per_replica="$LONG_CONTEXT_CONCURRENCY_PER_REPLICA"
      TASK_ARGS=(
        --max-seq-len 262144
        --max-out-len 32768
        --timeout 7200
        --stream-idle-timeout 1200
        --input-truncation-mode token_mid
      )
      ;;
  esac

  [[ "$MAX_DATASET_WORKERS" =~ ^[1-9][0-9]*$ ]] || die \
    'OPENCOMPASS_MAX_DATASET_WORKERS must be a positive integer'
  if (( dataset_workers > MAX_DATASET_WORKERS )); then
    dataset_workers="$MAX_DATASET_WORKERS"
  fi

  total_concurrency=$((VLLM_REPLICA_COUNT * concurrency_per_replica))
  workers_per_dataset=$(((total_concurrency + dataset_workers - 1) / dataset_workers))
  if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
    [[ "$BATCH_SIZE_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || die \
      'OPENCOMPASS_BATCH_SIZE_OVERRIDE must be a positive integer'
    request_batch_size="$BATCH_SIZE_OVERRIDE"
  else
    request_batch_size="$workers_per_dataset"
  fi
  TASK_ARGS+=(
    --batch-size "$request_batch_size"
    --max-workers "$workers_per_dataset"
    --query-per-second "$workers_per_dataset"
    --dataset-workers "$dataset_workers"
  )
  CONCURRENCY_DESCRIPTION=(
    "$concurrency_profile"
    "$VLLM_REPLICA_COUNT"
    "$concurrency_per_replica"
    "$total_concurrency"
    "$dataset_workers"
    "$workers_per_dataset"
    "$request_batch_size"
  )

  core_target="${CORE_TARGETS[$benchmark]:-}"
  [[ -n "$core_target" ]] || die \
    "internal error: no execution target for $benchmark"
  TASK_COMMAND=(
    bash "$SCRIPT_DIR/run_posttrain_objective_benchmark.sh"
    "$core_target"
    --work-dir "$work_dir"
    --reuse "$RUN_ID"
    "${COMMON_ARGS[@]}"
    # Safety limits for genuinely long-context tasks must win over global
    # concurrency requested for the regular suite.
    "${TASK_ARGS[@]}"
  )
}

verify_numeric_summary() {
  local summary_dir="$1"
  "$VERIFY_PYTHON" - "$summary_dir" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary_dir = Path(sys.argv[1])
summaries = sorted(summary_dir.glob('summary_*.csv'),
                   key=lambda path: (path.stat().st_mtime_ns, path.name),
                   reverse=True)
for summary in summaries:
    with summary.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    numeric = []
    for row in rows:
        if not row:
            continue
        value = str(list(row.values())[-1]).strip()
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            numeric.append((row.get('dataset', '<unknown>'), score))
    if numeric:
        preview = ', '.join(f'{name}={score:g}'
                            for name, score in numeric[:6])
        if len(numeric) > 6:
            preview += f', ... ({len(numeric)} scored rows)'
        print(f'Verified score summary: {summary} [{preview}]')
        raise SystemExit(0)
print(f'No numeric summary CSV found under {summary_dir}', file=sys.stderr)
raise SystemExit(1)
PY
}

build_agent_eval_command() {
  local benchmark="$1"
  local run_dir="$2"
  local system_name="${AGENT_EVAL_SYSTEM:-$MODEL_NAME}"
  AGENT_EVAL_COMMAND=(
    "$VERIFY_PYTHON" "$SCRIPT_DIR/opencompass_agent_eval_adapter.py"
    --benchmark "$benchmark"
    --run-dir "$run_dir"
    --model "$MODEL_NAME"
    --system "$system_name"
    --scope "$AGENT_EVAL_SCOPE"
    --platform-url "$AGENT_EVAL_PLATFORM_URL"
    --evidence-url "$AGENT_EVAL_EVIDENCE_URL"
    --provenance "$AGENT_EVAL_PROVENANCE"
    --publisher "$AGENT_EVAL_PUBLISHER"
    --mode "$AGENT_EVAL_MODE"
    --token-env "$AGENT_EVAL_TOKEN_ENV"
  )
  if [[ -n "$AGENT_EVAL_EVIDENCE_ROOT" ]]; then
    AGENT_EVAL_COMMAND+=(--evidence-root "$AGENT_EVAL_EVIDENCE_ROOT")
  fi
  if [[ -n "$AGENT_EVAL_CHECKPOINT_ID" ]]; then
    AGENT_EVAL_COMMAND+=(--model-checkpoint-id "$AGENT_EVAL_CHECKPOINT_ID")
  fi
  if [[ "$AGENT_EVAL_MODE" == ingest ]]; then
    AGENT_EVAL_COMMAND+=(
      --ingest-token-env "$AGENT_EVAL_INGEST_TOKEN_ENV"
    )
  fi
  if [[ -n "$AGENT_EVAL_MAPPING_FILE" ]]; then
    AGENT_EVAL_COMMAND+=(--mapping-file "$AGENT_EVAL_MAPPING_FILE")
  fi
  if (( AGENT_EVAL_DRY_RUN )); then
    AGENT_EVAL_COMMAND+=(--dry-run)
  fi
}

echo "Run id: $RUN_ID"
echo "Output root: $OUTPUT_ROOT"
echo "Selected benchmarks (${#SELECTED[@]}): ${SELECTED[*]}"
echo "Execution policy: sequential infer -> eval -> numeric-summary check"
if (( AGENT_EVAL_UPLOAD )); then
  if [[ "$AGENT_EVAL_DRY_RUN" == 1 ]]; then
    echo "Agent Eval: score -> server dry-run -> validate-only ($AGENT_EVAL_MODE mode)"
  elif [[ "$AGENT_EVAL_MODE" == ingest ]]; then
    echo "Agent Eval: score -> server dry-run -> direct canonical ingest (no Feishu review)"
  else
    echo "Agent Eval: score -> server dry-run -> pending-review upload"
  fi
fi

failures=()
for index in "${!SELECTED[@]}"; do
  benchmark="${SELECTED[$index]}"
  work_dir="$OUTPUT_ROOT/$benchmark"
  summary_dir="$work_dir/$RUN_ID/summary"
  build_command "$benchmark" "$work_dir"

  echo
  echo "=== [$((index + 1))/${#SELECTED[@]}] $benchmark: infer -> eval -> summary ==="
  echo "Concurrency: profile=${CONCURRENCY_DESCRIPTION[0]} "\
"replicas=${CONCURRENCY_DESCRIPTION[1]} "\
"per_replica=${CONCURRENCY_DESCRIPTION[2]} "\
"total=${CONCURRENCY_DESCRIPTION[3]} "\
"dataset_workers=${CONCURRENCY_DESCRIPTION[4]} "\
"workers_per_dataset=${CONCURRENCY_DESCRIPTION[5]} "\
"batch_size=${CONCURRENCY_DESCRIPTION[6]}"
  if (( DRY_RUN )); then
    printf 'DRY-RUN:'
    print_command_redacted "${TASK_COMMAND[@]}"
    echo "DRY-RUN score gate: $summary_dir/summary_*.csv must contain a numeric score"
    if (( AGENT_EVAL_UPLOAD )); then
      build_agent_eval_command "$benchmark" "$work_dir/$RUN_ID"
      printf 'DRY-RUN post-score upload:'
      print_command_redacted "${AGENT_EVAL_COMMAND[@]}"
    fi
    continue
  fi

  if "${TASK_COMMAND[@]}"; then
    if verify_numeric_summary "$summary_dir"; then
      if (( AGENT_EVAL_UPLOAD )); then
        build_agent_eval_command "$benchmark" "$work_dir/$RUN_ID"
        if "${AGENT_EVAL_COMMAND[@]}"; then
          if (( AGENT_EVAL_DRY_RUN )); then
            echo "COMPLETED: $benchmark (scored; Agent Eval submission validated only)"
          elif [[ "$AGENT_EVAL_MODE" == ingest ]]; then
            echo "COMPLETED: $benchmark (scored and canonically ingested into Agent Eval)"
          else
            echo "COMPLETED: $benchmark (scored and submitted to Agent Eval)"
          fi
          continue
        else
          status=$?
          echo "FAILED: $benchmark was scored, but Agent Eval submission exited with status $status." >&2
        fi
      else
        echo "COMPLETED: $benchmark"
        continue
      fi
    else
      status=1
      echo "FAILED: $benchmark inference/evaluation exited successfully but no numeric score was produced." >&2
    fi
  else
    status=$?
    echo "FAILED: $benchmark launcher exited with status $status." >&2
  fi

  failures+=("$benchmark:$status")
  if (( ! KEEP_GOING )); then
    echo "Stopping before the next benchmark; use --keep-going to continue after failures." >&2
    exit "$status"
  fi
done

if (( DRY_RUN )); then
  echo
  echo "Dry run complete: ${#SELECTED[@]} benchmark commands planned."
  exit 0
fi

if (( ${#failures[@]} > 0 )); then
  echo "Suite completed with failures: ${failures[*]}" >&2
  exit 1
fi

echo
echo "Suite complete: all ${#SELECTED[@]} selected benchmarks were inferred and scored."
