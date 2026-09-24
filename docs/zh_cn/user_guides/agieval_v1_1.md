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

v1.1 使用独立解析器，不修改旧版 AGIEval 或其他 benchmark 的公共解析规则。
仅处理最终 content，不把 reasoning_content 拼进答案；不发起二阶段请求或 LLM 判分。

### 选择题

1. 优先读取“最终答案 / final answer”，其次读取“答案是 / 应选 / 应选择 /
   Answer / Answer Choices”等明确结论。相同优先级取最后一个符合答案形式的标记，
   避免把“选项分析”“选择沉默”等普通叙述当作结论。
2. 支持 Markdown 粗体、引用、括号以及 LaTeX boxed、行内或跨行公式；
   支持选项后的数值或解释，如 `应选 **(A) 0.8**`。
3. 高考数学 351 题统一使用 agieval_mathqa_postprocess，支持 A–D 的连续字母、
   空格、逗号、顿号、列表和“和/及/与/and”。标准答案与预测均去重排序，
   通过 AccEvaluator 完整匹配；少选、多选均错，不给部分分。
4. 其他选择题使用 agieval_single_choice_postprocess，选项范围 A–E。
   没有明确答案时，先检查末尾纯选项或公式，再按文本顺序提取第一个大写选项字母保底。
   明确输出 AB 等多个选项时判无效，不取其中第一个字母。
5. 明确结论中的 `A 或 D`、`A or D`、`A/D` 等歧义表达判无效。
   高考数学没有从整段分析收集字母或取首字母的保底策略。

代码：opencompass/datasets/agieval/agieval_v1_1_postprocess.py。

### 填空题

math 与 gaokao-mathcloze 使用 AGIEvalV11ClozeEvaluator：

- 优先处理明确答案段；支持带嵌套花括号的 boxed/fbox 和常见数学公式包装。
  没有明确答案段时，读取末尾相邻的 boxed 答案组或数学公式组。
- 识别同一答案组中的多个空，按原顺序提取。参考答案用分号分隔；
  顶层分隔符可拆分多个值，坐标、区间、分式内部不拆分。
  不使用标准答案的内容或空数指导预测提取。
- 两侧均去掉 `$…$` 等包装，规范字体、百分号和变量赋值格式，
  保留 `2x+y+1=0`、`y=2x` 等完整方程。支持文字填空和带单位的数值回答。
- 规范化后沿用 is_equiv 的字符串比较。多个空数量、顺序和每一项都匹配才得分；
  不做符号代数求解，不采用数值容差，不以部分匹配判对。

代码：opencompass/datasets/agieval/agieval_v1_1_cloze.py。
这些规则扩展了旧版格式处理，不等同于官方原始评分脚本。
表达式虽然数学等价但书写不同，或自然语言结论无法被规则识别时，仍可能判错。

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
  tests/datasets/test_agieval_v1_1_extraction.py \
  tests/datasets/test_agieval.py \
  tests/datasets/test_run_benchmark_launcher.py -q
~~~
