# 新闻覆盖、行业映射与融合诊断

使用 `C:\Users\marti\anaconda3\envs\stockmixer\python.exe`，在 StockMixer 项目根目录运行：

```powershell
python scripts/16_check_news_pipeline.py
```

脚本读取现有 `configs/stockmixer.yaml` 与 `configs/llm.yaml`，不改配置。每次结果保存在独立的 `outputs/artifacts/news/diagnostics/年月日_时分秒_NASDAQ_NYSE_SP500_checks`。`REPORT.md` 是总报告，CSV/JSONL/JSON 是可复核明细。旧结果不覆盖。

可分步运行：

```powershell
python scripts/16_check_news_pipeline.py --step mapping
python scripts/16_check_news_pipeline.py --step coverage
python scripts/16_check_news_pipeline.py --step fusion
```

`coverage` 会先检查映射。上游全部完成并通过验收后，用以下命令要求全量统计：

```powershell
python scripts/16_check_news_pipeline.py --require-complete
```

## 1. 冻结及覆盖统计

- 上游重处理运行时，仅复制已完成并已导出的文章、审核及证据到本次隔离目录。逐文件记录 SHA256，并在读取结束后再次校验。文件变化则中止，重新运行即可。
- 已推理但未到导出检查点的新闻不混入。报告分别记录1948篇总量、已处理数和本次已导出冻结数；部分批次不代表全库分布。
- 适配、英文文本、市场路由证据及行业传播审核沿用下游现有校验函数，不额外调用 Qwen。
- 根据实际交易日收盘时刻、下一交易日规则、新闻保留窗口、历史关系有效期、字典标准/版本/代码和相同文本去重计算覆盖。统计以预测目标日划分训练、验证、测试。
- `coverage.csv` 每通道包含名义三划分及 `original_train_candidates`。后者保留原始代码训练候选中16个验证标签的重叠，不将其称为无泄露评估。
- `mapped_unique_articles` 是编码前映射篇数。`valid_price_label_cells` 还检查完整行情输入窗口及预测标签是否有效；共享市场文章按实际股票数广播计算格子，文章本身不会重复计篇。
- 新快照未编码时，`fresh_finbert_ready_articles=0` 表示本次未生成，不能解释为新闻永远不可用。候选篇数超过 PCA 维度仍需真实向量的秩检验。
- `coverage_verified_historical_only.csv` 排除无日期参考行业成员，便于查看真实历史证据造成的覆盖差距。
- `industry_link_candidate_stocks.csv` 逐行业新闻比较有日期成员及参考成员的候选股票数量；具体日期能否使用以覆盖台账为准。
- 各数据集 `*_mapped_news_ledger.jsonl` 保存来源文章ID、通道、股票或全池广播标记、决策日、目标日、划分和文本哈希。

## 2. 行业映射核验

校验三个数据集的完整股票轴、ticker、stock_uid、array_index，验证字典代码、标准、版本、名称及哈希，核对 CSV/JSONL/行业反向表一致性。真实 SEC-SIC 历史观察必须有原始证据、有效期和身份绑定。无日期参考成员会重新从原始行业压缩包及 SP500 CSV 生成并逐行比较，不依赖展示表自我声明。

有来源的无日期参考标签仅用于 `reference_retrospective` 探索实验；不等同于历史真实成员，不自动转成 observed PIT 数据。未匹配股票保存在 `unmatched_stock_industry.csv`，不会猜测补齐。

## 3. 融合功能诊断

- 只在 CPU 的临时内存权重上执行3步优化，使用现有配置学习率和原始损失函数，不启动正式训练，不保存或覆盖模型权重。
- 检查主干冻结、启用分支参数更新、关闭分支无更新、无新闻时严格等于基础预测、无效位置 NaN 不污染结果，以及推理重复一致性。
- Head 为零初始化时，第一步 Adapter/Gate 梯度为0正常；后续必须出现有效梯度，详见 `fusion_gradients.csv`。
- 第一个探针使用真实行情和临时随机新闻，验证功能路径；第二个探针读取旧 NASDAQ 实验真实 FinBERT 面板，验证真实特征能改变预测。旧面板只用于诊断，不复用到新重处理快照。
- 当前登记可训练参数包含未启用分支，因此分别报告登记参数量及实际启用参数量。
- 对保留的 `20261118_103625_NASDAQ_fusion` 中断实验只读检查历史记录和最后权重，明确标记“未封存诊断证据”，不会绕过正式权重加载器用于训练。
- 对共享市场向量，市场分支为 `r_new = r_base + A_day * sigmoid(B_day + w*r_base)`。导数下界 `1 - abs(A_day*w)/4` 大于0时，数学上仍保持基础排序。这可能导致 loss 变化而选股结果不变。浮点数舍入及原评估器的掩码处理可能产生细小变化。
- 原始评估器 RIC 是日度 IC 均值除以日度 IC 标准差，不是 RankIC。

功能诊断通过只说明数据路由与优化路径工作，不代表新闻模块已经带来预测提升。全量快照完成后，仍须正式导入、真实 FinBERT 编码、面板构建和匹配的基线/融合实验。
