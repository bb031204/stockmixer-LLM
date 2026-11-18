# 最终新闻输入验收与语义抽检

本次只修改下游新闻准备和验收。StockMixer主干、损失函数、训练采样、参数、评价函数及已有E0权重保持不变；运行中的上游v5批次保持不变。

## 已接入正式准备流程的验收

`main.py` 的 `validate` 阶段及训练入口解析已准备输入时，都会核验：

1. 新闻面板与真实FinBERT产物、股票轴、交易日轴和实验划分一致。
2. 面板中使用的文章全部编码成功，文本版本一致。mask与来源台账的股票位置、决策日、文章权重及重复文本相符。
3. 使用实际FinBERT向量重放市场事件聚类，核对事件ID；同一市场事件的多篇报道不会算成多个独立事件。公司/行业通道保留现有按文本去重逻辑。
4. 重新核验证据来源筛选；格式正确、有逐字引文仍不代表传播范围合理，语义问题由独立抽检记录限制。
5. 使用真实行情数组检查完整16日输入及下一日标签的有效性。最终新闻篇数、事件数、活跃天数及股票日格子均在这层过滤后统计。
6. 在名义训练期重建PCA拟合文章集合并重新拟合，核对保存的均值、投影矩阵、文章数及来源ID哈希；验证/测试文章不得参与拟合。
7. 对过滤后的市场事件数、有效天数执行现有 `market_channel.gates`，对启用通道检查 `minimum_training_cells`。不为通过验收降低门槛。

`readiness.json` 原有 `coverage` 为原始面板mask统计；新增 `final_inputs.coverage` 为最终可执行输入统计，`final_inputs_verified` 表示是否完成这层核验，`ready` 合并全部阻断项。

名义train/valid/test按预测目标日划分。原始StockMixer训练候选额外包含16个验证目标，单列为 `original_train_candidates`，不将它们当作名义训练覆盖或PCA拟合数据。

## 独立验收脚本

解释器为 `C:\Users\marti\anaconda3\envs\stockmixer\python.exe`。在StockMixer项目根目录，上游全量完成且验收通过后运行：

```powershell
python scripts/17_accept_news_features.py --datasets NASDAQ NYSE SP500
```

脚本冻结新快照、编码、构建关系、审核市场路由、检查实际覆盖和PCA。合格时构建面板；不激活正式快照，不训练模型。结果写入 `outputs/artifacts/news/acceptance/年月日_时分秒_news_acceptance/`。

上游未完成时可显式加 `--allow-partial` 做隔离诊断。部分结果始终不能标为正式就绪。没有足够训练文章或向量秩时，保留过滤后的输入候选统计及阻断原因，不发布空面板或自动切换降维方法。

主要文件：

- `FINAL_ACCEPTANCE_REPORT.md`：汇总结论和逐数据集阻断项。
- `acceptance.json`：输入快照、编码成功数、抽检状态及全部验收细节。
- `final_input_coverage.csv`：三个通道在各划分的最终输入统计。
- `<DATASET>_final_inputs.json`：事件数、活跃日及价格有效性检查结果。
- `SEMANTIC_SAMPLE_REPORT.md`：本次22篇新闻的逐条语义复核。
- `semantic_sample_review/reviews.jsonl`：可重放的来源绑定抽检限制。

完成正式导入后，原有 `python main.py --step all` 会自动使用增强验收；也可以按原有离线步骤准备，再运行 `src/train_strict.py`。

## 抽检与配置

`configs/llm.yaml` 新增唯一参数：

```yaml
pipeline:
  sample_review: outputs/artifacts/news/acceptance/20261118_180203_NASDAQ_NYSE_SP500_partial/semantic_sample_review
```

该文件夹由本轮保存的源记录版本、文本哈希和分类声明共同绑定。它只允许排除通道或某个已存在的行业关联，不允许添加行业、股票或新市场路由。来源或分类变化则拒绝复用旧结论。没有进入抽检样本的新闻不自动视为已审核。

本次抽检22篇，行业8、市场8、公司6；行业和市场为风险定向选择，公司为固定哈希顺序选择。12篇限制了部分传播，包含1篇只删除错误的发电行业关联、保留半导体关联。该比例不能作为全库错误率。

审核方法明确为 `assistant_secondary_review`，审核者Codex，人工确认数0。即使引文已与原文绑定，仍未保留完整原文来全面核验摘要所有事实，因此不宣称已完成人工全审或全文事实核验。

## 上游下一版本的建议

当前运行批次不改规则。若建设v6，应将本次抽检反例加入提示词和回归集：

- 个股退市/挂牌不合规不等于整个Nasdaq市场事件。
- 公司招聘、融资、重组、研究报告销售不自动构成同业共同冲击。
- `power semiconductor`不等于`Power Generation`。
- 行业名称、网页导航、机构成员名单、泛化风险免责声明不单独构成行业/全市场影响证据。
- 多行业分别提供传导证据，不能把同一泛化引文套到所有行业。
- 增补可复核的原文正文或带位置的上下文段落；仅保存引文和哈希不足以全面检查摘要是否失真。

这些改进应使用新批次、新提示词版本及新产物，保留当前版本可复现性。
