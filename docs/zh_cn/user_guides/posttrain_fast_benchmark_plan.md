# Post-train 快速评测与 vLLM 资源规划

本文面向后训练裸模型的快速回归。目标不是在每次 checkpoint 后跑完所有已适配
benchmark，而是在固定时间内覆盖最关键的能力维度，并保持不同模型之间可复现、
可比较。

## 1. 现有实跑结论

当前全量任务已经说明：单纯把客户端并发提高到 512，不能解决长思考输出造成的
decode 吞吐瓶颈。

| Benchmark | 规模 | 已观察耗时/状态 | 快速回归判断 |
| --- | ---: | --- | --- |
| MMLU-Pro | 12,032 | 4B、256 workers：4 小时 38 分 | 排除 |
| C-Eval | 1,346 | 4B、256 workers：56 分 41 秒 | 适合 |
| SuperGPQA Full | 26,529 | 4B 运行 78 分钟仅完成 719，约 2.7% | 排除 |
| MMLU-Redux | 5,330 | 2B non-thinking 约 3 分钟；thinking 成本显著增加 | 适合 |
| IFBench | 300 | 2B thinking、64 workers：约 49 分钟 | 适合 |
| AIME 2025 | 30 | 2B 历史有效 run：约 16 分钟 | 适合 |
| IFEval | 541 | 历史 full run 约 114 分钟且有补跑 | IFBench 的备选 |
| LiveCodeBench | 175 | 历史 full run 接近 10 小时且空答案多 | 仅抽样诊断 |
| LongBench v2 | 503 | 262K context，需要约 32 并发分批恢复 | 排除 |
| INCLUDE | 22,639 | full run 约 96 分钟 | 单独的多语言扩展项 |

本轮 4B thinking 输出中：

- C-Eval 每题平均约 12,183 个 reasoning 字符；
- MMLU-Pro 每题平均约 14,078 个 reasoning 字符；
- 两者 reasoning 字符的 P95 分别约 28,117 和 30,623。

因此 `max_out_len=32768` 会让少量长尾请求持续占用 KV cache，并拉长整项任务的
收尾时间。快速回归应把输出上限控制在 8192；需要对齐长思考官方结果时，再单独运行
16K/32K 配置。

## 2. 按能力维度选择 benchmark

| 能力维度 | 快速代表项 | 规模 | 选择理由 | 暂不选择的替代项 |
| --- | --- | ---: | --- | --- |
| 英文通识与综合推理 | MMLU-Redux | 5,330 | 覆盖面广，规模明显小于 MMLU-ProX Full | MMLU-Pro、MMLU-ProX Full |
| 中文知识与 STEM | C-Eval | 1,346 | 本轮 4B 已验证约一小时可完成 | SuperGPQA Full |
| 指令遵循 | IFBench | 300 | 规模小、区分度高、比 IFEval 更快 | IFEval |
| 数学推理 | AIME 2025 | 30 | 题量小，能快速发现推理能力退化 | AIME 2024/2026、HMMT |
| 专家科学推理 | GPQA Diamond | 792（198×4） | 能力价值高，但重复采样和长思考成本较高 | 放到扩展套件 |
| 代码生成 | LiveCodeBench sample 20 | 20 | 只做回归预警 | Full 175 不适合快速门禁 |
| 多语言 | MMMLU diagnostic 280 | 280 | 覆盖多语言方向 | MMMLU Full 196,588 |
| 长上下文 | 不进入快速套件 | — | LongBench v2/AA-LCR 对 KV 和时间影响过大 | 独立专项运行 |

LiveCodeBench sample 20 和 MMMLU diagnostic 280 都是内部诊断分数，不能与官方全量
结果直接比较。平台上传时必须明确标记 diagnostic protocol/sample count。

## 3. 推荐的三级评测门禁

### 3.1 P0：最快出首批结果

```text
ceval,ifbench,aime_2025
```

覆盖中文知识/STEM、指令遵循和数学推理，共 1,676 个请求。基于已有耗时上界，
thinking 模式约两小时；独占 vLLM、8192 输出上限并提高并发后，目标为 60–120 分钟。

适用场景：

- checkpoint 刚产出，需要尽快判断是否明显退化；
- 先给训练团队返回第一批可解释结果；
- GPU 资源暂时有限。

### 3.2 P1：推荐的代表性套件

```text
mmlu_redux,ceval,ifbench,aime_2025
```

共 7,006 个请求，覆盖英文通识、中文知识/STEM、指令遵循和数学推理。这是推荐固定
用于 2B/4B/5B checkpoint 横向比较的套件。

- non-thinking、8192 输出上限：目标 30–90 分钟；
- thinking、8192 输出上限：目标 1.5–3 小时；
- thinking、16K/32K：不再承诺两小时，应当进入 P2。

如果必须严格限制在两小时内，先跑 P0；P0 完成后再追加 MMLU-Redux。这样即使资源
不足，也能先得到三个能力维度的完整分数。

### 3.3 P2：发布候选或夜间扩展

在 P1 基础上按需求增加：

```text
gpqa_diamond
ifeval
include
livecodebench
```

不建议加入快速套件：

```text
mmlu_pro,supergpqa,mmmlu(full),mmlu_prox(full),global_piqa(full),
longbench_v2,aa_lcr,hmmt_feb_2026,hmmt_feb_2025,hmmt_nov_2025
```

## 4. 每类任务的建议客户端并发

`max_workers` 是 OpenCompass 同时在途请求上限，不等于 vLLM 实际同一步处理的序列数。
实际运行数还受 KV cache、`max_num_seqs` 和 token scheduler 限制。

| Benchmark | 建议 max_workers | 建议 QPS | max_out_len | 说明 |
| --- | ---: | ---: | ---: | --- |
| MMLU-Redux | 512 | 128–256 | 8192 | 题量大，适合高并发 |
| C-Eval | 512 | 128 | 8192 | 选择题；注意各 subject 的任务调度方式 |
| IFBench | 300 | 64–128 | 8192 | 总题量只有 300，超过 300 无收益 |
| AIME 2025 | 30 | 30 | 8192 | 总题量只有 30，增加到 512 无收益 |
| GPQA Diamond | 128–256 | 64 | 8192–16384 | 长思考，保留更多 KV 余量 |
| LiveCodeBench | 32–64 | 16–32 | 8192–16384 | 长生成和执行判题 |
| INCLUDE | 128–256 | 64–128 | 固定短输出 | `/v1/completions` prompt-logprob，不走 chat sampling |
| LongBench v2/AA-LCR | 16–32 | 8–16 | 32768 | 262K context，必须独立部署/运行 |

P1 使用同一个全局 `--max-workers 512` 是可行的：IFBench 和 AIME 会自然受自身题量
限制，不会真的制造 512 个请求。`--query-per-second 128` 已足以在平均请求时长超过
4 秒时维持约 512 个在途请求，不需要把 QPS 也设置成 512。

## 5. 两小时目标需要的真实吞吐

P1 总请求数为：

```text
5330 + 1346 + 300 + 30 = 7006
```

两小时完成所需平均请求吞吐：

```text
7006 / 7200 = 0.973 request/s
```

但资源规划应使用 output token/s，而不是只看 request/s：

| 模式假设 | 平均输出 | 两小时最低持续输出吞吐 | 建议带 30% 余量 |
| --- | ---: | ---: | ---: |
| non-thinking | 1,000 token/题 | 973 token/s | 1,300 token/s |
| 中等 thinking | 3,000 token/题 | 2,919 token/s | 4,000 token/s |
| 长 thinking | 8,000 token/题 | 7,784 token/s | 10,000 token/s |

这里还没有计算 prompt prefill，因此 4,000 output token/s 是 P1 thinking-8K 的推荐部署
验收目标，而不是理论最低值。

P0 只有 1,676 个请求。按平均 3,000 输出 token 估算，两小时最低约 698 token/s，
带余量后按 1,000 output token/s 验收即可。

## 6. 从单副本实测换算 GPU 副本数

在目标模型、目标量化精度和真实 benchmark prompt 上，先测一个 vLLM 副本的持续
generation token/s，记为 `T_replica`：

```text
P1 thinking 所需副本数 = ceil(4000 / T_replica)
P1 non-thinking 所需副本数 = ceil(1300 / T_replica)
P0 thinking 所需副本数 = ceil(1000 / T_replica)
```

例子：

| 单副本实测输出吞吐 | P0 thinking | P1 non-thinking | P1 thinking |
| ---: | ---: | ---: | ---: |
| 1,000 token/s | 1 副本 | 2 副本 | 4 副本 |
| 2,000 token/s | 1 副本 | 1 副本 | 2 副本 |
| 4,000 token/s | 1 副本 | 1 副本 | 1 副本 |

副本数是吞吐规划值，不代表模型权重本身需要这么多 GPU。对于能放进单卡的 2B/4B/5B
模型，应优先使用 TP=1 的独立数据并行副本扩展总吞吐；Tensor Parallel 主要用于模型
或目标 KV cache 无法放进单卡的情况。

## 7. vLLM 初始部署建议

以下仅作为 2B/4B/5B 文本模型的吞吐测试起点，最终值必须按实际模型结构、dtype、
GPU 型号和 vLLM 版本校准：

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len 65536 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

说明：

- `max_num_seqs=512` 是每个 data-parallel rank 的 scheduler 上限，不是保证 512 条
  65,536-token 序列可以同时驻留 KV cache；
- 小模型在大显存 GPU 上可以从 `max_num_batched_tokens=16384` 开始测试；
- 如果模型单卡可放下但持续 output tok/s 不足，优先增加 DP 副本；
- 若启用 vLLM 内部 DP，可从 `--data-parallel-size N --tensor-parallel-size 1` 开始；
- 若现有网关支持多个独立 endpoint，也可以启动 N 个独立 vLLM 实例，由网关负载均衡；
- 不要在没有实测时承诺“1 张卡支持 512 个 65K 请求”。最坏情况下 KV 需求远超权重
  占用。

## 8. GPU 显存档位的初步判断

在没有模型 `config.json`、KV dtype 和 GPU 型号时，不能给出精确卡数。可以先用下表
做部署方向判断：

| 单副本 GPU 显存 | 初始策略 | 风险 |
| --- | --- | --- |
| 24 GB | TP=1、active 64–128 起测，必要时多副本 | 权重能放下不代表 512 长序列 KV 能放下 |
| 48 GB | TP=1、active 128–256 起测 | thinking 长尾可能使 KV 接近满载 |
| 80 GB | TP=1、active 256 起测，再逐步升到 512 | 推荐作为快速评测单副本基线 |

原始 BF16 权重可粗略按 `参数量 × 2 bytes` 估算，但实际还包括运行时 workspace、
CUDA graph、激活和 KV cache。Qwen3.5 的精确 KV 成本必须用实际配置中的 layer、
KV-head、head-dim 和 KV dtype 计算，不能只用“5B 大约 10 GB 权重”推导并发。

## 9. vLLM 验收指标

压测时从 vLLM `/metrics` 观察：

```bash
curl -fsS "http://VLLM_HOST:PORT/metrics" | \
  rg 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions|generation_tokens)'
```

建议验收线：

| 指标 | 建议 |
| --- | --- |
| `num_requests_running` | 能接近所选 benchmark 的目标并发 |
| `num_requests_waiting` | 可以短时存在，但不能持续单调增长 |
| `kv_cache_usage_perc` | 稳态尽量低于 0.85，避免长期贴近 1.0 |
| `num_preemptions` | 稳态不持续增加 |
| generation token/s | P1 thinking ≥4,000；P0 thinking ≥1,000 |
| 超时/空响应 | 不能随并发增加显著上升 |

若 waiting 很高但 KV 使用率不高，检查 `max_num_seqs`、token scheduler、API 限流和
CPU/网络；若 KV 接近满载且 preemption 增长，应降低 active concurrency、缩短输出，
或增加副本/显存。

## 10. 推荐执行命令

### P0：优先在两小时内返回结果

```bash
cd /data2/agentic_benchmark_framework/opencompass/opencompass

bash script/run_benchmarks.sh \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --benchmark ceval,ifbench,aime_2025 \
  --max-seq-len 65536 \
  --max-out-len 8192 \
  --temperature 1.0 \
  --batch-size 1024 \
  --max-workers 512 \
  --query-per-second 128 \
  --extra-body-json '{"top_k":20,"min_p":0.0,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0}' \
  --output-root outputs/posttrain_p0 \
  --keep-going
```

### P1：推荐的完整快速回归

```bash
cd /data2/agentic_benchmark_framework/opencompass/opencompass

bash script/run_benchmarks.sh \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --benchmark mmlu_redux,ceval,ifbench,aime_2025 \
  --max-seq-len 65536 \
  --max-out-len 8192 \
  --temperature 1.0 \
  --batch-size 1024 \
  --max-workers 512 \
  --query-per-second 128 \
  --extra-body-json '{"top_k":20,"min_p":0.0,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0}' \
  --output-root outputs/posttrain_p1 \
  --keep-going
```

如果需要明确的 non-thinking 门禁，在 `extra-body-json` 中加入：

```json
"chat_template_kwargs":{"enable_thinking":false}
```

thinking 与 non-thinking 必须作为两个独立配置记录，不能把两种分数直接混在同一条
趋势线上。当前 launcher 对 benchmark 串行执行，每项内部并发；每个 benchmark 完成
推理后会立即评分，不会等所有推理完成后再统一评分。

## 11. 最终资源建议

在尚未拿到单副本 token/s 前，建议按下面方式申请资源：

1. 最小方案：1 个 80 GB GPU 副本，先保障 P0；
2. 推荐方案：2 个 TP=1 数据并行副本，目标覆盖 P1 thinking-8K；
3. 高保障方案：4 个 TP=1 副本，用于单副本仅约 1,000 token/s 或需要并行评测多个
   checkpoint 的情况；
4. LongBench v2、AA-LCR 和 full LiveCodeBench 使用独立资源池，不与快速门禁共用
   两小时 SLA。

最终卡数必须用 10–15 分钟真实 prompt 压测决定。只要测得单副本持续 output tok/s，
即可用第 6 节公式把 GPU 副本数定下来。

## 12. 32K thinking + 长上下文的两小时启动入口

如果被测模型只有在 reasoning 完成后才把最终答案写入 `content`，8192 的输出上限可能
截断最终答案。此时使用 `run_posttrain_2h.sh`：短阶段和长阶段都显式使用 32768 输出
上限，短阶段高并发，LongBench v2 降低到 32 个客户端并发；每个 benchmark 推理后
立即评分。

```bash
cd /data2/agentic_benchmark_framework/opencompass/opencompass

bash script/run_posttrain_2h.sh \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --output-root outputs/posttrain_2h
```

默认顺序是 `AIME 2025 -> IFBench -> C-Eval -> LongBench v2`。总预算为 7200 秒，
短阶段最多使用 3000 秒，为 LongBench v2 保留 4200 秒。预算耗尽后保留已有预测；使用输出中的同一个 `--run-id`
重新执行即可断点续跑。完整 LongBench v2 默认不抽样；`--long-samples N` 只用于诊断，
其结果不能作为官方全量分数上传。

C-Eval 的 52 个小 subject 默认用 16 个 dataset workers 并行，以便让总在途请求接近
512；其他三项保持 1 个 dataset worker，避免把“partition 并发 × 每个 partition 的 API
并发”意外放大。若服务端 KV cache 或排队压力过高，可用
`--ceval-dataset-workers 8 --short-workers 256` 下调。
