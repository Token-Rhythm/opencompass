# 语言 Benchmark 第二阶段适配

本页记录 PolyMath、MultiChallenge 和 AA-LCR 的 OpenCompass 适配与复现实验协议。
数据和代码来源均固定到不可变 revision，避免上游更新造成静默分数漂移。

## 已适配数据集

| Benchmark | 固定版本 | OpenCompass 配置 | 官方协议对齐点 |
|---|---|---|---|
| PolyMath | `Qwen/PolyMath@71e0db902e0ece05e208dfd8b1695bd3d95cf130` | `polymath_0shot_gen` | 18 种语言 × 4 个难度 × 125 题；官方本地化 boxed-answer instruction；提取第一个 `boxed{}`；数学等价判分；各语言按 1/2/4/8 加权 |
| MultiChallenge | `ekwinox117/multi-challenge@5ccefcca6a39020d66c1383c4e6a809cb07afa33` | `multichallenge_gen` | 保留完整多轮历史；官方 judge prompt；`gpt-4o-2024-08-06`、temperature 0、结构化 YES/NO；先算四个 axis，再做宏平均 |
| AA-LCR | `ArtificialAnalysis/AA-LCR@bdae010bbce259820c0e34c1d7cce210d966fb75` | `aa_lcr_gen` | 100 道长上下文题；按 CSV 中的文件顺序拼接文档；官方 prompt 与 equality-checker prompt；官方 Qwen3 judge |

PolyMath 主榜分数可使用：

```bash
python run.py \
  --models <model-config> \
  --datasets polymath_0shot_gen \
  --summarizer polymath
```

该配置的 `polymath_<lang>` 是四个难度 accuracy 的官方加权结果：
`(low + 2*medium + 4*high + 8*top) / 15`；最外层 `polymath` 是 18
种语言的宏平均。官方还报告 thinking/answer language consistency，但这要求推理服务把
thinking 和 final answer 分成两个字段；OpenCompass 的通用生成结果只有一个 response，
因此本适配只复现主榜 weighted accuracy，不伪造这两个辅助指标。

## LLM Judge 的严格复现条件

MultiChallenge 和 AA-LCR 的分数依赖指定 judge，不能随意替换模型。两个配置已经写入
官方 judge 模型名、temperature 和 judge prompt，并通过 OpenAI-compatible API 调用。
运行前设置：

```bash
export OPENAI_API_KEY=<key>
export OPENAI_BASE_URL=<openai-compatible-base-url>
```

分别运行：

```bash
python run.py --models <model-config> --datasets multichallenge_gen
python run.py --models <model-config> --datasets aa_lcr_gen
```

- MultiChallenge 必须让服务端的 `gpt-4o-2024-08-06` 对应 OpenAI 同名快照，且支持
  `response_format=json_schema`。配置与官方一样使用 4096 个 judge 输出 token。
- AA-LCR 必须让 `Qwen/Qwen3-235B-A22B-Instruct-2507` 对应官方指定的
  Qwen3 235B A22B 2507 non-reasoning 模型。不要换成 thinking 版本，也不要让服务端
  自动启用 thinking。
- 如果服务端暴露的模型别名不同，只覆盖配置中的 `path`；实际模型权重、版本和推理模式
  必须保持不变，否则结果不能与官网直接比较。
- AA-LCR 的每条输入平均约 100k token。被测模型和 judge 服务都必须支持至少 131072
  token；不应通过截断来跑出一个看似完整的分数。

## 依赖与缓存

PolyMath 数学等价判分使用 OpenCompass extra requirements 中的 `sympy`、
`antlr4-python3-runtime==4.11` 和 `latex2sympy2_extended`。如当前环境是最小安装：

```bash
uv pip install -r requirements/extra.txt
```

三个数据集首次运行时会从官方 Hugging Face 或 GitHub 地址下载并缓存数据。

## 本阶段跳过的数据集

以下项目没有注册空壳配置，也没有用第三方数据或相似 judge 冒充官方协议：

| Benchmark | 跳过原因 | 可继续适配的条件 |
|---|---|---|
| MaXIFE | 官方论文已公开，但未找到官方公开的数据集和评分代码 | 官方发布数据、完整 prompt 和 scorer |
| NOVA-63 | 官方 Hugging Face 仓库目前只有说明页，数据/代码仍标记为待发布 | 官方仓库实际发布数据和 evaluator |
| WMT24++ | WMT 数据可从 `mt-metrics-eval` 获取，但论文主结果所用的 MetricX-24 evaluator 权重未按原版本公开；公开模型说明自身只是最接近官方提交的版本 | 发布论文所用 evaluator 权重，或官方明确给出可复现替代协议 |
| BFCL-V4 | 用户明确排除 | 重新纳入范围 |
| TAU2-Bench | 用户明确排除 | 重新纳入范围 |

这些限制是为了保证“能跑”与“能复现官网分数”不会被混为一谈。
