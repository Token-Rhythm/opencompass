# OpenCompass benchmark downsampling

## Scope

The production OpenCompass runner evaluates 15 benchmark families. The exact
pre-downsampling inference-row count is 596,871.

| Benchmark | Original rows | Production rows | Production name |
| --- | ---: | ---: | --- |
| C-Eval | 1,346 | 1,346 | `ceval` |
| IFEval | 541 | 541 | `ifeval` |
| MMMLU | 196,588 | 1,400 | `mmmlu_downsampling` |
| GPQA Diamond | 792 | 792 | `gpqa_diamond` |
| IFBench | 300 | 300 | `ifbench` |
| AIME 2025 | 30 | 30 | `aime_2025` |
| AIME 2026 | 120 | 120 | `aime_2026` |
| LiveCodeBench | 175 | 175 | `livecodebench` |
| MMLU-Redux 2.0 | 5,330 | 1,400 | `mmlu_redux_downsampling` |
| MMLU-ProX | 341,011 | 1,400 | `mmlu_prox_downsampling` |
| Global PIQA | 27,091 | 1,400 | `global_piqa_downsampling` |
| INCLUDE base-44 | 22,639 | 1,400 | `include_downsampling` |
| MultiChallenge | 273 | 273 | `multichallenge` |
| HMMT February 2026 | 132 | 132 | `hmmt_feb_2026` |
| LongBench v2 | 503 | 503 | `longbench_v2` |

The resulting production suite contains 11,212 inference rows: 4,212 rows
from the ten unchanged benchmarks and 7,000 sampled rows from the five large
benchmarks.

## Sampling contract

- The manifest is generated deterministically with seed `20260801`.
- Every large benchmark receives a budget of 1,400 rows.
- Allocation is proportional to the original stratum sizes, uses largest
  remainder rounding, and includes at least one row from every non-empty
  stratum.
- Sampling is without replacement and stores explicit source row indices.
- MMMLU is stratified by language and subject (798 strata).
- MMLU-ProX is stratified by language and category (406 strata).
- MMLU-Redux is stratified by subject (57 strata).
- Global PIQA is stratified by variant and language (267 strata).
- INCLUDE is stratified by language (44 strata).

The checked-in manifest is authoritative. Regeneration with unchanged source
data and seed must produce the same indices.

## Runtime profile

The 1,400-row budget is based on an observed generation throughput of about
49.5 rows per minute with two vLLM replicas and a runner-wide maximum of 512
request workers. This gives an estimated 28.3 minutes for each sampled
generation benchmark. Prompt-logprob benchmarks can finish faster. The target
is per large benchmark, not 30 minutes for the entire 15-benchmark suite.

Regular tasks use `batch_size=1024`, `max_workers=512`, and QPS 128. HMMT and
LongBench v2 retain their long-context safety profiles. The deploy-side vLLM
instances use a 256K model length and 256 maximum sequences.
