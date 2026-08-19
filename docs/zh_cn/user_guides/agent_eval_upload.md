# OpenCompass 评测结果直传 Agent Eval Hub

本文说明如何把已经完成评分的 OpenCompass benchmark 结果，通过 Agent Eval Hub
的 canonical ingest API 直接写入测评展示平台。

本文使用的生产地址为：

```text
https://47.88.93.207
```

这条链路不经过飞书、不创建审核任务，也不需要人工审批。`dry_run=false` 成功后，
Evidence、Eval Run 和 Eval Result 会直接写入平台的 canonical 数据库，网页随后即可
读取。

需要注意：“不经过飞书审核”不等于“API 无鉴权”。当前生产接口仍要求两个相互独立
的凭证：

- `AGENT_EVAL_SUBMISSION_PAT`：只用于读取可提交的 canonical Catalog，从而解析
  System、BenchmarkVersion 和 Metric ID；
- `INGEST_ADMIN_TOKEN`：只用于调用 `POST /api/ingest` 直接落库。

不要把两个 Token 写进脚本、JSON、Git、飞书或命令行参数。本文中的命令均从环境
变量读取 Token。

## 1. 完整数据流

每个 benchmark 使用一条独立事务：

```text
OpenCompass inference
  -> OpenCompass eval
  -> summary_*.csv
  -> 读取 canonical Catalog 并解析 ID
  -> 生成 evidence + run + aggregate result payload
  -> POST /api/ingest，dry_run=true
  -> 检查校验结果
  -> POST /api/ingest，dry_run=false
  -> 保存回执中的 run/result ID
  -> 网页展示
```

一次上传不会触发飞书同步。平台若设置了 `FEISHU_OUTBOX_ENABLED=0`，也不会创建新的
飞书 Outbox 记录。

## 2. 上传前必须准备的内容

### 2.1 已完成的 OpenCompass run 目录

run 目录至少应包含：

```text
<run-dir>/
├── configs/*.py
├── predictions/**
└── summary/summary_*.csv
```

首先确认 summary 存在并且包含数值分数：

```bash
RUN_DIR='/data2/agentic_benchmark_framework/opencompass/opencompass/outputs/你的输出目录/benchmark/run_id'

find "$RUN_DIR/summary" -maxdepth 1 -name 'summary_*.csv' -type f -print
```

再查看内容：

```bash
sed -n '1,20p' "$RUN_DIR/summary/summary_xxx.csv"
```

没有数值 summary 的任务不能上传为已完成结果。

### 2.2 平台中已经注册的实体

上传结果之前，平台中必须已经存在：

- 与本次模型对应的 `System`，且 scope 正确；
- 对应的 `Benchmark` 和精确 `BenchmarkVersion/Split`；
- 对应的 `Metric`；
- 如果填写 `model_checkpoint_id`，该 Checkpoint 必须属于 System 对应的 Model。

`POST /api/ingest` 不会自动创建这些目录实体，也不会用模糊名称猜测版本。

后训练裸模型一般使用：

```text
scope = posttrained
System = 纯模型，不带 OpenSquilla Harness
```

模型加 OpenSquilla 的 Agent 评测一般使用：

```text
scope = agent_system
System = Model + 精确 Harness 版本
```

不要把裸模型结果上传到 `agent_system`，也不要把带 Harness 的结果上传到
`posttrained`。

### 2.3 长期可访问的 Evidence URL

平台只保存证据 URI，不负责上传本地文件。因此 Evidence 必须是能够长期访问的
HTTP/HTTPS 地址，例如由 Nginx 或对象存储提供的 summary CSV：

```text
https://artifacts.example/evals/{benchmark}/{run_id}/{summary_file}
```

可以上传 CSV 对应的 URL，但不能填写：

```text
/data2/.../summary.csv
file:///data2/.../summary.csv
```

推荐 URL 使用下列占位符：

- `{benchmark}`：统一 benchmark 名称；
- `{run_id}`：OpenCompass run ID；
- `{summary_file}`：summary 文件名；
- `{summary_sha256}`：summary 内容哈希。

例如：

```bash
export AGENT_EVAL_EVIDENCE_URL='https://artifacts.example/evals/{benchmark}/{run_id}/{summary_file}'
```

## 3. 准备凭证和检查平台

在评测机的交互式 shell 中读取 Token，避免写入 shell history：

```bash
read -rsp 'Catalog PAT: ' AGENT_EVAL_SUBMISSION_PAT; echo
export AGENT_EVAL_SUBMISSION_PAT

read -rsp 'Ingest admin token: ' INGEST_ADMIN_TOKEN; echo
export INGEST_ADMIN_TOKEN
```

设置平台地址：

```bash
export AGENT_EVAL_PLATFORM_URL='https://47.88.93.207'
```

检查服务健康：

```bash
curl -fsS "${AGENT_EVAL_PLATFORM_URL}/api/healthz" | python3 -m json.tool
```

预期响应：

```json
{
  "ok": true
}
```

查询最小提交目录：

```bash
CATALOG_FILE='/data2/liyulong/tmp/agent-eval-upload-catalog.json'

curl -fsS \
  -H "Authorization: Bearer ${AGENT_EVAL_SUBMISSION_PAT}" \
  -o "$CATALOG_FILE" \
  "${AGENT_EVAL_PLATFORM_URL}/api/submissions/catalog"

chmod 600 "$CATALOG_FILE"
python3 -m json.tool "$CATALOG_FILE" >/dev/null
wc -c "$CATALOG_FILE"
```

Catalog 的 `source` 必须是 `d1`，不能是演示数据：

```bash
python3 - "$CATALOG_FILE" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding='utf-8'))
assert document.get('source') == 'd1', document.get('source')
for key in ('systems', 'benchmarks', 'benchmarkVersions', 'metrics'):
    assert isinstance(document.get('data', {}).get(key), list), key
print('Catalog OK')
PY
```

如果这里返回 401，检查 `AGENT_EVAL_SUBMISSION_PAT`。如果返回 403，Token 缺少
`submission:create` 权限或数据范围不包含本次 scope。

## 4. 推荐方法：由适配器生成 canonical payload

适配器会读取 summary、prediction 数量和保存的 OpenCompass config，并自动完成：

- 选择该 benchmark 的主指标；
- 精确匹配 System、BenchmarkVersion 和 Metric；
- 根据 Metric unit 把 `85.72` 保持为 percent，或转换为 `0.8572` ratio；
- 提取样本数、开始/结束时间和总耗时；
- 记录代码 commit、config hash、summary hash 和数据集版本；
- 记录 `temperature`、`top_p`、`top_k`、`min_p`、`seed`、输出上限；
- 把 `presence_penalty`、`repetition_penalty` 等放入 `sampling_config`；
- 只在配置明确写出时记录 `thinking_enabled`；
- 排除 API key、Authorization、Cookie、password、secret 和 token。

INCLUDE 使用 `/v1/completions` prompt-logprob 协议，因此适配器不会为 INCLUDE 登记
没有实际使用的 chat generation 采样参数。

### 4.1 设置本次结果变量

以下以一个已经完成的 C-Eval run 为例，替换成真实值：

```bash
cd /data2/agentic_benchmark_framework/opencompass/opencompass

BENCHMARK='ceval'
MODEL='你的模型服务名'
SYSTEM='平台中已经注册的精确 System 名称或 sys_ ID'
RUN_DIR='/data2/agentic_benchmark_framework/opencompass/opencompass/outputs/你的输出目录/ceval/你的run_id'
EVIDENCE_URL='https://artifacts.example/evals/{benchmark}/{run_id}/{summary_file}'
CATALOG_FILE='/data2/liyulong/tmp/agent-eval-upload-catalog.json'
```

适配器支持的 benchmark 名称是：

```text
mmlu_pro, ceval, supergpqa, ifeval, mmmlu, gpqa_diamond, ifbench,
longbench_v2, aime_2024, aime_2025, aime_2026, hmmt_feb_2026,
livecodebench, mmlu_redux, mmlu_prox, global_piqa,
hmmt_feb_2025, hmmt_nov_2025, include, aa_lcr
```

### 4.2 只生成 payload，不写平台

```bash
../opencompass-venv/bin/python script/opencompass_agent_eval_adapter.py \
  --benchmark "$BENCHMARK" \
  --run-dir "$RUN_DIR" \
  --model "$MODEL" \
  --system "$SYSTEM" \
  --scope posttrained \
  --platform-url "$AGENT_EVAL_PLATFORM_URL" \
  --evidence-url "$EVIDENCE_URL" \
  --provenance internal_private \
  --publisher 'OpenCompass Evaluation Team' \
  --mode ingest \
  --catalog-file "$CATALOG_FILE" \
  --payload-only
```

生成文件：

```text
<run-dir>/agent_eval/<benchmark>.ingest.json
```

它是一个本地回执包装，真正要发送的 API 正文位于 `.payload`。

设置文件路径并检查关键字段：

```bash
RECEIPT_FILE="$RUN_DIR/agent_eval/$BENCHMARK.ingest.json"

python3 - "$RECEIPT_FILE" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding='utf-8'))
payload = record['payload']
run = payload['runs'][0]
result = payload['results'][0]
print('idempotency_key:', payload['idempotency_key'])
print('system_id:', run['system_id'])
print('benchmark_version_id:', run['benchmark_version_id'])
print('metric_id:', result['metric_id'])
print('score:', result['value'])
print('sample_count:', result.get('sample_count'))
print('temperature:', run.get('temperature'))
print('top_p:', run.get('top_p'))
print('max_output_tokens:', run.get('max_output_tokens'))
print('thinking_enabled:', run.get('thinking_enabled', '<unknown>'))
print('evidence_url:', payload['evidence'][0]['url'])
PY
```

在上传前人工确认：模型/System、benchmark 版本、split、指标、单位、分数、样本数和
Evidence URL 都正确。

## 5. 先执行 API dry-run

从适配器回执中提取 `.payload`，只把 `dry_run` 设置为 `true`：

```bash
DRY_PAYLOAD='/data2/liyulong/tmp/agent-eval-ingest-dry.json'

python3 - "$RECEIPT_FILE" "$DRY_PAYLOAD" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding='utf-8'))
payload = record['payload']
payload['dry_run'] = True
with open(sys.argv[2], 'w', encoding='utf-8') as output:
    json.dump(payload, output, ensure_ascii=False, indent=2)
    output.write('\n')
PY

chmod 600 "$DRY_PAYLOAD"
```

调用 API：

```bash
DRY_RECEIPT='/data2/liyulong/tmp/agent-eval-ingest-dry-receipt.json'

curl -fsS \
  -X POST \
  -H "Authorization: Bearer ${INGEST_ADMIN_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data-binary "@$DRY_PAYLOAD" \
  -o "$DRY_RECEIPT" \
  "${AGENT_EVAL_PLATFORM_URL}/api/ingest"

chmod 600 "$DRY_RECEIPT"
python3 -m json.tool "$DRY_RECEIPT"
```

成功响应必须包含：

```json
{
  "ok": true,
  "dry_run": true,
  "validation_mode": "schema_only",
  "normalized": {}
}
```

`schema_only` 的含义是：结构、枚举、数值范围和批内引用通过，但这一阶段不会查询
数据库验证 `system_id`、`benchmark_version_id`、`metric_id` 是否存在。适配器此前的
Catalog 精确解析负责降低这类风险；正式 apply 时服务端仍会再次验证目录引用。

如果 dry-run 返回 422，查看响应中的 `issues[]`，按其中的 `path` 修正 payload。
不要跳过 dry-run。

## 6. 正式写入 canonical 数据

只有 dry-run 成功并完成字段复核后，才生成 apply 正文。除了 `dry_run`，不要修改
任何字段，也不要更换 `idempotency_key`：

```bash
APPLY_PAYLOAD='/data2/liyulong/tmp/agent-eval-ingest-apply.json'

python3 - "$RECEIPT_FILE" "$APPLY_PAYLOAD" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding='utf-8'))
payload = record['payload']
payload['dry_run'] = False
with open(sys.argv[2], 'w', encoding='utf-8') as output:
    json.dump(payload, output, ensure_ascii=False, indent=2)
    output.write('\n')
PY

chmod 600 "$APPLY_PAYLOAD"
```

正式提交：

```bash
APPLY_RECEIPT='/data2/liyulong/tmp/agent-eval-ingest-apply-receipt.json'

curl -fsS \
  -X POST \
  -H "Authorization: Bearer ${INGEST_ADMIN_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data-binary "@$APPLY_PAYLOAD" \
  -o "$APPLY_RECEIPT" \
  "${AGENT_EVAL_PLATFORM_URL}/api/ingest"

chmod 600 "$APPLY_RECEIPT"
python3 -m json.tool "$APPLY_RECEIPT"
```

首次写入成功通常返回 HTTP 201，正文类似：

```json
{
  "ok": true,
  "dry_run": false,
  "idempotency_key": "opencompass....",
  "request_hash": "...",
  "replayed": false,
  "created": {
    "evidence": 1,
    "runs": 1,
    "samples": 0,
    "results": 1
  },
  "ids": {
    "evidence": ["ev_..."],
    "runs": ["run_..."],
    "samples": [],
    "results": ["res_..."]
  }
}
```

必须保存 `request_hash`、`ids.runs[]` 和 `ids.results[]`。它们是排查网页展示和审计
记录的最直接依据。

## 7. 幂等重试规则

平台使用 `idempotency_key + request_hash` 防止重复写入：

- 同一个 key、完全相同的正文：安全重放，返回 HTTP 200 和 `replayed=true`；
- 同一个 key、不同正文：返回 HTTP 409 `idempotency_conflict`；
- 网络超时、429、502、503、504：使用原正文和原 key 有界重试；
- 不要因为超时生成随机新 key，否则可能创建重复 Run；
- 真正修改了分数、协议、版本、System 或采样配置：生成表达新语义的新 key，不能
  覆盖旧 Run。

如果 apply 请求超时，不要猜测是否写入成功，直接原样重发同一个
`$APPLY_PAYLOAD`。若第一次已成功，第二次会返回 `replayed=true`。

## 8. 网页验收

上传成功后不需要执行飞书同步。使用浏览器登录平台：

```text
https://47.88.93.207/
```

依次检查：

1. “最近运行记录”中出现 `ids.runs[0]` 对应的 Run；
2. System 名称和 scope 为 `posttrained`；
3. Benchmark version/split 正确；
4. 分数和 Metric unit 正确；
5. Checkpoint、temperature、top_p、top_k、min_p、max tokens 和 thinking 配置正确；
6. Evidence 链接可以打开；
7. 对比报告中出现该 System、配置和结果。

网页最近运行区域只展示有限数量的最新 Run。未出现在最近列表中不等于数据丢失，
还应通过对比报告或 Dashboard API 按 `ids.runs[0]` 检查。

`INGEST_ADMIN_TOKEN` 不能读取 Dashboard。程序化验收需要另外持有
`dashboard:read` PAT：

```bash
curl -fsS \
  -H "Authorization: Bearer ${AGENT_EVAL_DASHBOARD_PAT}" \
  "${AGENT_EVAL_PLATFORM_URL}/api/dashboard" \
  > /data2/liyulong/tmp/agent-eval-dashboard.json
```

## 9. 直接手写 `/api/ingest` payload

不使用 OpenCompass adapter 时，可以直接构造下面的最小 aggregate 结果。所有
`REPLACE_*` 都必须替换成 canonical Catalog 中的真实值：

```json
{
  "idempotency_key": "opencompass.ceval.system-revision.run-20260729.accuracy",
  "dry_run": true,
  "source": "opencompass",
  "evidence": [
    {
      "local_id": "evidence",
      "provenance": "internal_private",
      "title": "OpenCompass C-Eval run 20260729_120000",
      "publisher": "OpenCompass Evaluation Team",
      "url": "https://artifacts.example/evals/ceval/20260729_120000/summary.csv",
      "retrieved_at": "2026-07-29T13:00:00+08:00",
      "is_open": false,
      "content_hash": "sha256:REPLACE_WITH_SUMMARY_SHA256"
    }
  ],
  "runs": [
    {
      "local_id": "run",
      "system_id": "sys_REPLACE",
      "benchmark_version_id": "bmv_REPLACE",
      "evidence_local_id": "evidence",
      "scope": "posttrained",
      "status": "completed",
      "runner_version": "opencompass@REPLACE",
      "code_commit_sha": "REPLACE_WITH_GIT_SHA",
      "dataset_snapshot": "REPLACE_WITH_DATASET_REVISION",
      "config_hash": "sha256:REPLACE_WITH_CONFIG_SHA256",
      "temperature": 1.0,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0,
      "max_output_tokens": 8192,
      "thinking_enabled": true,
      "sampling_config": {
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0
      },
      "sample_count_planned": 1346,
      "sample_count_completed": 1346,
      "started_at": "2026-07-29T12:00:00+08:00",
      "completed_at": "2026-07-29T13:00:00+08:00",
      "total_time_ms": 3600000
    }
  ],
  "results": [
    {
      "run_local_id": "run",
      "metric_id": "met_REPLACE",
      "evidence_local_id": "evidence",
      "result_level": "aggregate",
      "scorer_key": "accuracy",
      "value": 85.72,
      "sample_count": 1346,
      "is_primary": true
    }
  ]
}
```

Metric unit 不在 Result 中重复填写，而是由 `metric_id` 指向的 canonical Metric
决定：

- Metric unit 是 `percent`：85.72% 写 `85.72`；
- Metric unit 是 `ratio`：85.72% 写 `0.8572`。

单位写错会造成展示结果放大或缩小 100 倍，这是人工上传中最需要检查的字段。

可选的 `model_checkpoint_id` 应放在 `runs[0]`，且必须属于当前 System 对应的 Model。
省略表示此次 Run 没有绑定已注册 Checkpoint，不表示平台会自动选择最新
Checkpoint。

## 10. 字段与批次限制

单个 `/api/ingest` 原子批次最多包含：

| 数组 | 最大条数 | 用途 |
| --- | ---: | --- |
| `evidence` | 10 | 证据来源 |
| `runs` | 10 | 评测运行 |
| `samples` | 25 | 可选的样本级记录 |
| `results` | 40 | aggregate、task 或 sample 指标 |

OpenCompass 常规上传推荐“一次一个 benchmark、一个 Run、一个主 Result”，不要把逐题
prediction 全部塞进 API。大型输出、trace、输入和预测文件应放在对象存储或 Nginx，
平台只保存 URI 和摘要。

Run 的重要约束：

- `scope` 只能是 `pretrained`、`posttrained` 或 `agent_system`；
- `status=completed` 时必须填写 `completed_at`；
- `sample_count_completed` 不能大于 `sample_count_planned`；
- `working_time_ms` 不能大于 `total_time_ms`；
- `top_p`、`min_p` 范围为 0 到 1；
- `top_k`、token 数和样本数必须为非负整数；
- `thinking_enabled` 未知时应省略，不能猜成 `false`。

Result 的重要约束：

- `result_level` 为 `aggregate`、`task` 或 `sample`；
- aggregate 结果通常设置 `is_primary=true`；
- task 结果必须填写属于当前 BenchmarkVersion 的 `benchmark_task_id`；
- sample 结果必须引用同批次 `samples[].local_id`；
- 无效结果必须设置 `is_invalid=true` 并填写 `invalidation_reason` 或
  `error_code`。

## 11. 常见错误

### HTTP 401

```json
{"error":"ingest_service_authentication_required"}
```

原因：缺少或错误的 `INGEST_ADMIN_TOKEN`。

### HTTP 403

Catalog 查询 PAT 缺少权限或数据范围不包含当前 scope/provenance。

### HTTP 409

```json
{"error":"idempotency_conflict"}
```

同一幂等键已用于不同正文。不要自动换随机 key；先对比旧回执、System、版本、指标、
分数和配置，确认这是重试还是一次真正的新评测。

### HTTP 422 schema_validation_failed

请求结构、枚举、时间、数值范围或批内引用错误。逐条查看 `issues[].path`。

### HTTP 422 reference_validation_failed

System、Checkpoint、BenchmarkVersion、Metric、Task 或 Evidence ID 不存在，或者 scope、
Checkpoint Model、Task Version 不一致。重新下载 Catalog 并精确映射，不能伪造 ID。

### HTTP 503

数据库不可用或 ingest token 未配置。不要改幂等键；恢复平台后用原正文重试。

### 上传成功但网页没看到

依次检查：

1. apply 回执是否为 `ok=true`、`dry_run=false`；
2. 是否保存了 `ids.runs` 和 `ids.results`；
3. scope 是否选错；
4. System 是否是预期 System；
5. Metric unit/value 是否正确；
6. 页面是否只显示最近若干 Run；
7. 当前筛选条件是否排除了 provenance 或 System；
8. 是否把裸模型错误地上传成了带 Harness 的 Agent System。

## 12. 清理敏感环境变量

完成上传后清理当前 shell 中的凭证：

```bash
unset AGENT_EVAL_SUBMISSION_PAT
unset INGEST_ADMIN_TOKEN
unset AGENT_EVAL_DASHBOARD_PAT
```

本地 Catalog、payload 和回执不应包含 Token，但仍包含内部 System、结果、配置和证据
信息，应保持 `chmod 600`，并按团队的数据保留策略归档。
