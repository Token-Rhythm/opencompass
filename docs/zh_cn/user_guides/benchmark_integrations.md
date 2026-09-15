# 新增及扩展 Benchmark 适配说明

更新日期：2026-09-15。

本文汇总本仓库近期新增的数据集、已有 benchmark 的协议扩展，以及统一评测入口。
内容依据当前工作区代码（含未提交的 SciCode 相关修改）整理。
“已适配”表示已有数据加载、推理配置或评分实现，不代表本文记录了一次全量评测验收。

## 1. 适配范围

### 1.1 新增的数据集及协议配置

下表的“套件名称”用于 `script/run_benchmarks.sh --benchmark`。
“非默认套件”表示已有适配，但不能直接把该名称传给这一入口。

| Benchmark | 内容与协议 | 主要配置 | 套件名称 / 入口 |
| --- | --- | --- | --- |
| MMLU-Redux 2.0 | 57 个学科；筛选标注为 `ok` 的题目；生成式选择题评分 | `mmlu_redux_gen` | `mmlu_redux_downsampling` |
| MMLU-ProX | 29 种语言 × 14 类；Full/Lite，0-shot/5-shot CoT；本地化提示和答案提取 | `mmlu_prox_5shot_cot_gen` 等四套配置 | `mmlu_prox_downsampling`，使用 Full 5-shot CoT |
| Global PIQA | parallel/nonparallel 两种版本；各语言准确率先宏平均，再对两个版本宏平均 | `global_piqa_generation` | `global_piqa_downsampling` |
| INCLUDE base-44 | 44 种语言；对选项 continuation 求 log-likelihood，选择得分最高的选项 | `include_base_44_0shot_ppl` | `include_downsampling` |
| HMMT February / November 2025 | 两场数学竞赛；MathArena 提示与解析器；当前配置每题单次生成 | `hmmt_2025_matharena_gen` | `hmmt_feb_2025`、`hmmt_nov_2025` |
| AIME 2025 | 数学题生成；boxed 答案提示；使用 `MATHVerifyEvaluator` | `aime2025_gen` | `aime_2025` |
| AIME 2026 | MathArena 数据和评分；配置为每题重复 4 次 | `aime2026_gen` | `aime_2026` |
| HMMT February 2026 | MathArena 数据和评分；配置为每题重复 4 次 | `hmmt2026_gen` | `hmmt_feb_2026` |
| PolyMath | 18 种语言、4 档难度；数学等价评分；难度权重为 1/2/4/8 | `polymath_0shot_gen` | 非默认套件；通过 OpenCompass CLI 运行 |
| MultiChallenge | 保留多轮历史；指定 judge 评分；按 axis 汇总后宏平均 | `multichallenge_gen` | 非默认套件；共享核心支持 `multichallenge` |
| AA-LCR | 长上下文问答；按指定顺序拼接文档；指定 judge 判断答案 | `aa_lcr_gen` | 非默认套件；共享核心支持 `aa_lcr` |
| SciCode 官方带背景提示版本 | 按题并行、题内步骤顺序生成；官方提示和代码提取方式；本地沙箱执行测试 | `scicode_official_wbg_gen` | `scicode` |
| LiveCodeBench v6 | 固定 `test6.jsonl` 的 175 题；生成 Python 代码并执行测试 | `livecodebench_v6_codegen` | `livecodebench` |

MMLU-ProX 的四套配置分别是：

- `mmlu_prox_5shot_cot_gen`
- `mmlu_prox_0shot_cot_gen`
- `mmlu_prox_lite_5shot_cot_gen`
- `mmlu_prox_lite_0shot_cot_gen`

数据 revision、本地化提示来源和详细评分约定见
[第一阶段适配说明](language_benchmarks_phase1.md)、
[第二阶段适配说明](language_benchmarks_phase2.md)。

### 1.2 已有 benchmark 的扩展

这些项目不全部属于全新数据集，本轮工作还包含协议、数据加载、结果提取和运行接入：

| Benchmark | 当前扩展内容 |
| --- | --- |
| MMLU-Pro | 新增 5-shot CoT 配置及对应评分、汇总调整 |
| C-Eval | 新增 EvalScope 风格的单上下文 5-shot 配置；统一套件的 `ceval` 映射到此配置 |
| MMMLU | 数据加载和答案提取调整，接入固定降采样 |
| GPQA-Diamond、SuperGPQA | 提示、答案解析或评分协议调整；配有失败样本排查及重跑结果合并工具 |
| IFEval、IFBench | 加载和规则评分适配；IFBench 同时保留主指标与其他诊断指标 |
| LongBench v2 | 数据加载、上下文处理和长上下文运行参数调整 |
| AIME 2024、HumanEval | 复用现有数据集实现，纳入统一运行入口 |

## 2. 统一运行入口

主链路为：

```text
script/run_benchmarks.sh
  -> script/run_posttrain_objective_benchmark.sh
  -> 数据集配置 + VLLMOpenAIAPI
  -> OpenCompass 推理 -> 评分 -> 汇总
```

套件按顺序执行 benchmark；每项产生数值汇总 CSV 后才进入下一项。
当前默认选择以下 21 个运行项：

```text
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
```

以实际脚本中的 `ALL_BENCHMARKS` 和 `CORE_TARGETS` 为准。
这是运行项数量，不能直接当作“新增 benchmark 数量”。

在仓库根目录运行：

```bash
# 查看入口支持的 benchmark 名称
bash script/run_benchmarks.sh --list-benchmarks

# 仅打印执行计划；模型、端点、tokenizer 均需替换成实际值
bash script/run_benchmarks.sh \
  --model served-model-name \
  --base-url http://127.0.0.1:8000/v1 \
  --tokenizer-path /path/to/tokenizer \
  --benchmark aime_2025,mmlu_redux_downsampling \
  --dry-run
```

正式运行时移除 `--dry-run`；`--run-id` / `--reuse` 用于指定或复用运行目录。
默认输出父目录为 `outputs/benchmark_suite`。
通用的 `--samples N` 表示每个数据集配置的样本上限，不是整个套件的总样本数；
固定降采样任务已设置显式索引，不应再同时传入样本上限。

共享核心另外支持 AA-LCR、MultiChallenge 和部分多语言任务的全量入口。例如：

```bash
bash script/run_posttrain_objective_benchmark.sh multichallenge \
  --model served-model-name \
  --base-url http://127.0.0.1:8000/v1 \
  --tokenizer-path /path/to/tokenizer
```

此命令还需要预先配置 MultiChallenge 的 judge 服务，不能只准备被测模型。
PolyMath 可使用自定义模型配置运行：

```bash
python run.py --models your_model_config \
  --datasets polymath_0shot_gen --summarizer polymath
```

更多入口参数见脚本的 `--help`；
结果上传与目录映射见 [Agent Eval 上报说明](agent_eval_upload.md)。

## 3. 固定降采样

当前五个大规模任务各使用 1,400 条样本：

| 任务 | 分层依据 |
| --- | --- |
| MMMLU | 语言 × 学科 |
| MMLU-Redux | 学科 |
| MMLU-ProX | 语言 × 类别 |
| Global PIQA | 数据版本 × 语言 |
| INCLUDE | 语言 |

索引保存在
[`manifest.json`](../../../opencompass/configs/datasets/downsampling/manifest.json)，
使用固定种子 `20260801` 生成，以显式原始行索引选择样本。
它们是固定子集评测，不能直接标为对应 benchmark 的全量结果。

[降采样说明](../../../opencompass/configs/datasets/downsampling/README.md)
记录了生成方案和当时的 15 类生产套件统计；该历史统计不等于本文列出的当前 21 项默认套件。

## 4. 评分与复现注意事项

### 4.1 Chat、原始 completion 与 logprob

`VLLMOpenAIAPI` 支持 chat 生成、原始 completion 生成和 continuation log-likelihood。
统一运行核心对生成任务选择 chat；INCLUDE 走原始 `/v1/completions`，
通过 `prompt_logprobs` 和 `return_token_ids` 计算各选项续写 token 的对数概率之和。

MMLU-Redux、MMLU-ProX 等配置可用于原始 prompt 协议，但统一套件采用 chat。
要复现原始 completion 协议，应使用相应模型配置或支持端点切换的入口，
同时检查 chat template、few-shot、采样参数和停止条件。
提示正文相同并不意味着两种传输协议产生的分数可以直接等同。

当前推理链路保存 `reasoning_content` 与最终 `content`；
通用评分链路优先读取最终答案。PolyMath 当前适配只报告数学正确率汇总，
尚未实现其 thinking/answer language consistency 辅助指标。

### 4.2 数学重复采样

HMMT 2025 配置是单次生成；AIME 2026 和 HMMT 2026 配置显式设置
`num_repeats=4`。应区分原始题数和推理行数。
重复生成后的平均正确率也不等于“4 次中任一次答对即得分”的 pass@4。

采用 MathArena parser，并不自动意味着完整复现其外层运行流程；
比较成绩前还需核对重复次数、采样参数以及是否包含 last-chance 重报答案请求。

### 4.3 SciCode

`scicode` 当前选择官方带背景提示配置，结果名称为
`SciCode_official_with_background_sandboxed`：

- 不同题目可并行，同一道题的步骤按依赖顺序生成。
- 提示构造及代码提取对齐官方对应实现。
- 测试构造、执行隔离及默认 120 秒超时复用本地评分器。
- 当前工作区包含 SciPy `simps` 兼容、沙箱脚本只读绑定、基础设施错误识别、
  汇总模式识别以及短临时路径等修复。
- 分别报告 `accuracy` 和 `sub_accuracy`。

因此，该配置应标注为“官方提示 + 本地沙箱评分”，不能仅凭名称认定其与完整官方 runner 完全一致。

### 4.4 LLM Judge 与长上下文

MultiChallenge 的配置指定 `gpt-4o-2024-08-06`；
AA-LCR 的配置指定 `Qwen/Qwen3-235B-A22B-Instruct-2507`。
judge 的模型版本、推理模式、提示和结构化输出能力会影响结果，
需要与被测模型端点分别配置。

AA-LCR 和 LongBench v2 还需要记录实际上下文上限及截断策略。
尤其共享核心中的 AA-LCR 默认采用 `mid` 输入处理，
不能把存在输入截断的结果直接视作完整长上下文协议复现。
具体要求见 [第二阶段适配说明](language_benchmarks_phase2.md)。

## 5. 代码与验证入口

主要实现位置：

- [数据集目录](../../../opencompass/datasets)：loader、答案提取、评分器。
- [数据集配置](../../../opencompass/configs/datasets)：提示、few-shot、推理器和评分器组合。
- [汇总分组](../../../opencompass/configs/summarizers/groups)：语言、学科及加权汇总。
- [统一套件脚本](../../../script/run_benchmarks.sh)、
  [共享运行核心](../../../script/run_posttrain_objective_benchmark.sh)。
- [vLLM API 适配](../../../opencompass/models/vllm_openai_api.py)。
- [数据集测试目录](../../../tests/datasets)：协议、解析、固定索引及运行入口相关测试。

正式扩展或变更协议时，至少核对数据版本和样本数、实际 prompt、
标准答案字段、评分器、汇总范围，并通过小样本验证后再运行全量。
单元测试通过不能替代实际模型运行和结果覆盖率检查。

## 6. AGIEval v1.1

已新增 v1.1 本地数据加载、zero-shot / zero-shot-CoT 单阶段推理配置，
并接入共享 post 脚本的 agieval 入口，默认 zero-shot。
共 21 个任务、7,272 条记录；高考数学选择题中的 7 道多选题全部保留，
使用完整选项集合提取和完全匹配。JEC-QA 单元素列表标签归一化为单选字母。
本次适配不提供 few-shot，不自动纳入默认 21 项套件。

数据来源、评分细节、推理参数和启动命令见 [AGIEval v1.1 说明](agieval_v1_1.md)。
