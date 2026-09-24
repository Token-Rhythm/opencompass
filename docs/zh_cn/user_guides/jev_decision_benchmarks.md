# JevBench 与 Kev 决策评测

这两个项目在 OpenCompass 中运行。OpenJev 项目的静态评测与游戏实验统一在 RunWeaver 中维护，不在这里重复实现。

本次首先接通生成式 LLM。模型输出完整的概率 JSON，评分器保留概率、原始回答与异常。概率来源标记为 `verbalized`；它不是 token logprob，也不是 Jev 复现模型的分类头概率。尚未拿到复现模型，所以其原生接口适配和精度验证没有完成。

## 数据、版本和评分

源码提交、文件 SHA256 和许可保存在 `opencompass/datasets/_decision_upstream/SOURCES.json`。JevBench 的评分文件直接复用上游；Kev 只提取纯评分函数，函数正文不变，避免引入训练代码和额外模型依赖。

| 数据集 | 默认记录数 | 说明 |
|---|---:|---|
| JevBench public | 231 决策 | easy 48、original 72、hard 111。仓库没有榜单全部私有和导入题，不能称为 534 题全集 |
| Kev decision-v7 | 1204 条 development 记录 | 保留一条记录内的全部问题和变体 |
| Kev transfer-v4 | 764 条 development 记录 | 独立报告，不与其它 suite 合成一个分数 |
| Kev transfer-v9 | 1264 条 development 记录 | 包含证据不足的 unknowable 样本 |

题目首次使用时从固定提交下载，逐文件核对哈希，缓存支持并发锁与原子写入。可通过 `OPENCOMPASS_DECISION_CACHE` 指定共享缓存位置，也可用 `--dataset-kwargs-json '{"data_root":"/path/to/pinned/checkout"}'` 读取本地固定版本。

JevBench 保留上游 argmax、并列处理、概率和的严格及舍入容差、Brier、ECE、ordinal MAE。无效回答保留为错误，不编造概率。只报告公开题目的指标，不伪造私有榜单综合分、成本或延迟。

Kev 保留上游 clean/variant、选项排列、成对样本、NLL、Brier、ECE、风险覆盖率等报告。unknowable 的标签不能作为普通准确率真值；上游 clean 准确率将其排除。成对样本不完整或概率输出无效时，不发布假装完整的原生分数，记录原因。小样本切片可能打断配对关系，因此仅用于验证链路。

原生详细报告在 OpenCompass 结果 JSON 的 `native_report` 中；摘要显示准确率、Brier、ECE 和 schema 有效率。准确率和 schema 有效率使用百分数，Brier/ECE 保持原始尺度。风险覆盖率是当前样本内的经验统计，不是部署后错误率保证。

## 启动方式

沿用项目惯例：`script/run_jevbench.sh`、`script/run_kev.sh` 是薄入口，均调用统一 launcher。

```bash
bash script/run_jevbench.sh \
  --base-url http://MODEL_HOST:MODEL_PORT/v1 \
  --model MODEL_ID --tokenizer-path /path/to/tokenizer \
  --max-workers 4 --query-per-second 4 \
  --work-dir outputs/jevbench_my_model

bash script/run_kev.sh \
  --base-url http://MODEL_HOST:MODEL_PORT/v1 \
  --model MODEL_ID --tokenizer-path /path/to/tokenizer \
  --max-workers 4 --query-per-second 4 \
  --work-dir outputs/kev_my_model
```

默认生成温度 0、输出预算 4096。JevBench 的 system/user 文本和逐题严格 JSON schema 均取自上游 OpenAI adapter；并发请求各自持有 schema，不共享可变配置。服务不支持该约束时直接报错，不自动降级成普通 JSON。JevBench 和 Kev 的请求耗尽重试后会使推理失败，不把网络故障转成空回答计入模型错误。Kev 使用 JSON-object 输出约束；其上游原生接口为结构化决策接口，当前聊天提示词属于显式的生成式模型适配。Kev 使用一条记录一次请求的显式 JSON 模板，保留问题与选项顺序，不把标签和 `_meta` 发给模型。

可以用 `--samples 3` 做诊断。正式测评去掉该参数。控制思考模式等服务参数使用现有 `--extra-body-json`，推理参数会保存到运行配置。不要把 smoke 的设置和成绩当作正式基线。

Kev 默认只读取 development。需要最终 test 时使用 `--kev-partition test`，其摘要名称自动变为 `_test`；不可用 dataset kwargs 把 development 悄悄改成 test。调参时应保留 test，不能用 test 挑选模型或阈值。三个 suite 可能有共同来源，不能当成相互独立的样本集合相加。

平台导出适配已登记 `jevbench_public` 及六个 `kev_{decision_v7,transfer_v4,transfer_v9}_{dev,test}`，没有创建 Kev 总分。此次没有向外部平台发送数据。

## 已验证的范围（2026-09-22）

- 固定题目加载：231、1204、764、1264；标准答案回放，四组准确率均为 100%。
- 回归测试：概率舍入、并列、缺键、NaN、无效输出、人口不一致及原生评分边界。
- Flash-Next 实际服务：JevBench 3 题、Kev 每个 suite 3 条记录，通过正式 shell 入口完成推理、评分和摘要。
- 以上是真实小样本链路验证，不是正式全量评测，也不证明未知 Jev 复现模型已适配。

## 对齐边界

评分测试验证的是固定上游版本的算法和题目，并不意味着取得官方私有榜单结果。JevBench 原生 runner 串行测量延迟；当前并行运行的延迟不能直接对照其串行延迟指标，需要时设置并发为 1 并单独测量。OpenCompass 复用现有环境，未改动待测模型服务；环境警告不等于原生 Jev 权重路径已验证。正式评测须保存依赖版本、服务配置、采样设置和数据提交。

逐题 schema 的修正已通过 Flash-Next 实际服务的 3 题 smoke；旧的 JSON-object smoke 仅保留作开发记录。Kev test 配置已通过加载验证，三个 suite 分别为 1176、764、1264 条，未对 test 集进行模型推理。

## 8666 全量运行（2026-09-22）

用户已授权使用现有 Flash-Next 服务进行真实验证，并将可用任务扩展至全量 trial 1。配置、队列、结果统一记录在 [当前进度与结果](/data4/zhangshuo/Experiment/jev-integration/flash-next-8666-20260922/当前进度与结果.md)。小样本与正式结果使用独立目录。视觉路径已发送真实 Doom 画面并取得模型响应；完整 episode 及全集状态以该进度记录为准。
