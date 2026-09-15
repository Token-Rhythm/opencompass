# AGIEval v1.1 适配说明

## 数据版本与范围

官方仓库：https://github.com/ruixiangcui/AGIEval。
数据固定到提交 84ab72d94318290aad2e4ec820d535a95a1f7552 的 data/v1_1。

本服务器数据目录：

~~~
/data2/agentic_benchmark_framework/opencompass/opencompass/data/AGIEval/data/v1_1
~~~

21 个 JSONL 文件共 7,272 条记录。其中 sat-en-without-passage 是 206 条
SAT 英语题去掉 passage 的变体；排除该变体后为 20 个任务、7,066 条记录。
原始数据不改写；source_manifest.json 保存来源、commit、文件大小、题数与 SHA256。
loader 只读本地文件，不会在推理时下载数据。

高考数学选择题 gaokao-mathqa 共 351 题，其中 7 题有多个正确选项。
原始文件的 1-based 行号与标签如下：

| 行号 | 原始标签 | 加载后的标签 |
| --- | --- | --- |
| 149 | AD | AD |
| 150 | ACD | ACD |
| 246 | A B D | ABD |
| 247 | A C | AC |
| 248 | B C D | BCD |
| 286 | CD | CD |
| 287 | AC | AC |

这 7 题全部保留。JEC-QA 的单元素列表标签（例如 ["B"]）加载为 B。
当前下载版本的 JEC-QA 和高考物理正式测试题标签均为单选。

## 推理协议

仅支持以下两种模式，默认 zero-shot：

| 模式 | 配置名 | 提示构造 |
| --- | --- | --- |
| zero-shot | agieval_v1_1_gen | 复用 convert_zero_shot |
| zero-shot-CoT | agieval_v1_1_zeroshot_cot_gen | 复用 convert_zero_shot_CoT_stage1 |

两种模式均通过 chat 接口发送一条 system 消息和一条 user 消息；
system 为 "You are a helpful AI assistant."，user 内容由现有 AGIEval 提示构造器生成。
每题只进行一次生成，不进行官方第二阶段作答或 LLM 答案提取。
不额外追加“单选题”或其他答案格式提示。

本次 v1.1 适配不提供 few-shot、few-shot-CoT 或 chat_mode=False。
下载目录中的官方 few_shot_prompts.csv 仅保留为来源文件，不参与提示构造。
旧版 AGIEval 配置不受这次模式调整影响。

## 答案提取和匹配

### 高考数学选择题

所有 351 题统一使用 agieval_mathqa_postprocess，而不是根据标准答案决定提取方式。

1. 优先识别“最终答案 / 最后答案 / final answer”；没有时识别普通答案标记、
   “因此选 / 故选”等结论标记。相同优先级使用最后出现的标记。
2. 从标记后的答案行或答案句中解析选项；没有标记时支持末尾 boxed 答案
   或最后一行的纯选项。不会从整段分析中收集所有出现过的字母。
3. 支持连续字母、空格、逗号、顿号、括号、列表、和/及/与/and、
   常见 Markdown 或 LaTeX 包装。选项限 A–D；统一大小写、去重、排序。
4. 标准标签去除空白、去重、排序，再由 AccEvaluator 完全匹配。
   不给部分分；少选、多选均错。单选标准答案为 B 时，预测 AB 也判错。
5. 空输出、无法识别的输出、“A 或 D / A or D / A/D”等歧义表达提取为空，
   按错误计入分母，不丢弃样本。

例如标准答案为 AD 时，AD、A D、D，A、["A", "D"] 均正确；
A、ABD 则错误。带分析的“B 不正确，最终答案是 A、D。”提取为 AD。

数学选项支持换行的 \[...\]、\(...\) 和 $$...$$ 公式包装，以及其中的 \boxed{AD}。遇到答案标记时先读取完整公式块，再解析整个选项表达式，避免在公式首行截断。仍拒绝 A 或 D、少选、多选和公式后同句的歧义补充；不从分析过程搜集字母。

这是保守的规则解析，不是语义判分。最终答案中混入未支持的说明文字时可能提取失败；
可通过预测结果与评测 details 排查。规则以
opencompass/datasets/agieval/agieval_v1_1_postprocess.py 为准。

### 其他任务

- 其他选择题：复用 first_option_postprocess(options='ABCDE') 和 AccEvaluator。
- 两个填空任务 math、gaokao-mathcloze：复用 AGIEvalEvaluator、
  parse_math_answer 和 is_equiv。
- 评分沿用当前框架对最终 content 的处理，不将单独的 reasoning_content 拼入最终答案。

## 启动方式

在服务器仓库根目录运行，模型名、端点和 tokenizer 路径替换为实际值：

~~~bash
cd /data2/agentic_benchmark_framework/opencompass/opencompass

# 默认 zero-shot
bash script/run_posttrain_objective_benchmark.sh agieval \
  --model served-model-name \
  --base-url http://127.0.0.1:8000/v1 \
  --tokenizer-path /path/to/tokenizer

# zero-shot-CoT
bash script/run_posttrain_objective_benchmark.sh agieval \
  --agieval-setting zero-shot-CoT \
  --model served-model-name \
  --base-url http://127.0.0.1:8000/v1 \
  --tokenizer-path /path/to/tokenizer
~~~

agieval_v1_1 是 agieval 的别名。
可先添加 --samples 1 做每任务一题的小样本检查；
--dry-run 仅构造与划分任务，不代表完成真实模型推理。
此入口已接入共享 post 脚本；不自动加入 run_benchmarks.sh 的默认 21 项套件。

沿用共享推理默认值：

| 参数 | 默认值 |
| --- | --- |
| max_seq_len | 65536 |
| max_out_len | 4096 |
| temperature | 1.0 |
| batch_size | 512 |
| query_per_second | 64 |
| max_workers | 64 |
| retry | 1 |
| timeout | 3600 秒 |
| dataset_workers | 1 |

其余采样参数、extra_body、流式选项沿用共享脚本默认值。
--dataset-kwargs-json 可覆盖本地数据 path；不允许用它绕过模式参数校验。

默认输出目录分别为 outputs/agieval_v1_1_zero_shot_chat 和
outputs/agieval_v1_1_zero_shot_cot_chat，可通过 --work-dir 覆盖。

汇总报告全 21 个任务、英文选择题 8 个、中文组选择题 11 个、
数学填空题 2 个的宏平均。中文组沿用既有分类，包括 gaokao-english。
全任务宏平均包含 SAT 无 passage 变体；它不是去重后的逐题总准确率。

## 验证

2026-09-15 完成两种模式下全部 21 个任务、各 7,272 条记录的离线加载检查。
数学 7 道多选题全部保留，标准标签已规范化。

回归测试覆盖旧版 AGIEval、数学选项提取和完整匹配、单阶段生成消息、
模式移除、配置加载及共享启动入口。模拟生成验证每题一次请求，
不代表已经运行真实模型的全量评测。

~~~bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  ../opencompass-venv/bin/python -B -m pytest \
  tests/datasets/test_agieval_v1_1.py \
  tests/datasets/test_agieval.py \
  tests/datasets/test_run_benchmark_launcher.py -q
~~~
