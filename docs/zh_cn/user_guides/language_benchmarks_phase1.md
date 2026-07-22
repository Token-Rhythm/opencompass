# 语言 Benchmark 第一阶段适配

本页记录 MMLU-Redux 2.0、MMLU-ProX、Global PIQA、INCLUDE base-44、
HMMT February/November 2025 的可复现实验协议。所有 Hugging Face 数据源
均固定到不可变 revision；更新数据版本时应显式修改 revision 并重新做协议对齐，
不应静默跟随数据集主分支。

## 运行配置

```bash
opencompass --models <model-config> --datasets mmlu_redux_gen
opencompass --models <model-config> --datasets mmlu_prox_5shot_cot_gen --summarizer mmlu_prox
opencompass --models <model-config> --datasets global_piqa_generation
opencompass --models <model-config> --datasets include_base_44_0shot_ppl
opencompass --models <model-config> --datasets hmmt_2025_matharena_gen
```

HMMT 的官方 MathArena parser 依赖 extra requirements 中的
`antlr4-python3-runtime==4.11`、`sympy` 和 `regex`。parser 固定自 MathArena
commit `a11194deff8c67a232974a383795e8a2776b4c6f`，以避免安装完整、
Python 3.12-only 的 MathArena 工程。

通过 vLLM 的 OpenAI-compatible 服务运行 INCLUDE 时，模型配置必须使用
`VLLMOpenAIAPI`，并把 `openai_api_base` 指向 `/v1` 根路径。该模型类默认对普通生成
任务使用 `/v1/chat/completions`，但对 INCLUDE 使用 `/v1/completions` 的
`prompt_logprobs=0` 和 `return_token_ids=true`：分别计算 `" A"`、`" B"`、
`" C"`、`" D"` continuation 的全部 token log-probability 之和，再取最大值。
此流程不使用 chat template、生成答案或 LLM Judge。

```python
from opencompass.models import VLLMOpenAIAPI

models = [dict(
    type=VLLMOpenAIAPI,
    abbr='qwen3.5-9b-vllm',
    path='Qwen3.5-9B',
    key='EMPTY',
    openai_api_base='http://127.0.0.1:8000/v1',
    max_seq_len=65536,
)]
```

MMLU-Redux、MMLU-ProX 和 Global PIQA 的上述 lm-evaluation-harness 固定版本
使用原始 completion prompt；官方 MMLU-ProX 复现命令也没有传
`--apply_chat_template`。如果目标是对齐该协议，而不只是用指令模型做 chat
评测，应显式使用原始生成端点并且不要配置 `meta_template`：

```python
models = [dict(
    type=VLLMOpenAIAPI,
    abbr='qwen3.5-9b-vllm-raw',
    path='Qwen3.5-9B',
    key='EMPTY',
    openai_api_base='http://127.0.0.1:8000/v1',
    generation_endpoint='completions',
    meta_template=None,
    max_seq_len=131072,
)]
```

`generation_endpoint='completions'` 只改变生成任务的传输方式；INCLUDE 无论该
选项为何值，都使用原始 completions prompt logprob。若改用 chat endpoint，题面
文字仍相同，但 chat template 会增加 system/user/assistant token，所得分数不再是
上述 lm-evaluation-harness 原始协议的严格复现。

## 数据和协议固定点

| Benchmark | 官方数据 revision | 复现协议 |
|---|---|---|
| MMLU-Redux 2.0 | `edinburgh-dawg/mmlu-redux-2.0@372ea425445d51e1ba1188c56e5e893f8138621f` | 57 个 subject；仅保留 `error_type == "ok"`；官方 generative prompt；提取首个大写 A-D；按样本数聚合 |
| MMLU-ProX | `li-lab/MMLU-ProX@8e6106a6c6ce1c5027e66cc338143cf997b2aa09` | 29 种语言 × 14 类；每类 validation 前 5 条 CoT few-shot；本地化 instruction/prompt/答案正则；确定性生成 |
| Global PIQA nonparallel | `mrlbenchmarks/global-piqa-nonparallel@6777742fa3634c0583cda3b7f8a482ea7b1b0937` | 官方 generation prompt、严格答案正则、`temperature=0.8`、`top_p=0.95` |
| Global PIQA parallel | `mrlbenchmarks/global-piqa-parallel@b0b18516a8bc2cb1106bce3dd4db32848ca715ea` | 同上；先对各语言做宏平均，再对 parallel/nonparallel 做宏平均 |
| INCLUDE base-44 | `CohereLabs/include-base-44@d2e1f6015f67a43c02a9a68db98e2298e2d6a660` | 44 种语言；官方 zero-shot multiple-choice log-likelihood；每种语言单独报告 accuracy |
| HMMT Feb 2025 | `MathArena/hmmt_feb_2025@6fdc4277120810ff75aa22d2d5489b91f7a262a1` | 30 题；官方 instruction；MathArena `strict_parsing=false` 评分 |
| HMMT Nov 2025 | `MathArena/hmmt_nov_2025@118dbfb45c4c9467c672268ed55166642897aa46` | 30 题；协议同上 |

协议实现对齐到 lm-evaluation-harness commit
`f4d4b3de3ee6741a7151a9fe74945ee515262f4c`。MMLU-ProX 的本地化文案来自
该版本并保留 MIT license；HMMT parser 也保留 MathArena MIT license。

## NOVA-63 状态

截至本适配版本，官方 Hugging Face 仓库 `zjy1298/NOVA-63` 的固定 revision
`7bfc98546474aeabddee1f6fdfe1091a28bddbb0` 只有 README，没有数据文件；
官方页面同时标记 Dataset/Code 尚未发布。因此 OpenCompass 不注册空壳任务，
也不使用第三方镜像替代。官方发布数据和评分代码后，才能在不猜测协议的前提下
完成可复现适配。
