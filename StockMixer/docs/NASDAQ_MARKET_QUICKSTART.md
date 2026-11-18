# NASDAQ：原始 StockMixer 口径 + Market 单通道探索对照

更新：2026-11-18。本次只准备配置、脚本、离线面板和检查，不启动训练或预测。

## 当前选择

- 数据集：NASDAQ，1026 只股票。
- 单通道：Market，全市场新闻上下文，例如美联储、宏观经济、系统性金融风险。
- Company、Industry 分支暂不启用；保留已有结构和数据接口。
- 行情与训练口径：`original`，调用现有原始复现数据和随机候选窗口实现。
- 时间性质：`retrospective`，保留新闻实际处理时间，不声称历史已在线生成。
- 新闻证据：现阶段允许标题与未完成原文来源核验的摘要，因此是探索性消融，不是原文证据完整的正式验收。

## original 和 strict 的区别

`train_strict.py` 是兼容保留的入口名称，实际模式由配置决定。文件名不代表一定采用 strict。

| 项目 | original（本次） | strict（保留可选） |
|---|---|---|
| 行情预处理 | 保留原始数据尺度 | 使用训练期拟合的缩放 |
| NASDAQ 每轮训练 | 打乱起点 0～755，取 740 个 | 使用合法训练窗口 |
| 候选目标日期索引 | 16～771 | 16～755 |
| 验证标签重叠 | 可能使用索引 756～771 的验证标签 | 不使用这部分验证标签训练 |
| 实验解释 | 原始实现复现对照，不能称无泄漏 | 标签切分更严格；新闻另受其时间协议约束 |

本次 E0 与融合都使用 original，不能把融合结果直接与 strict 基线横比。新闻不会因启用 original 而获得未来发布日期，但原始训练窗口本身仍存在上述标签重叠；PCA 继续只在名义训练区间拟合，这不能消除标签重叠。

原始模型文件、原始损失、评价函数与训练循环保持原样。修改的是新闻流水线的协议选择、同协议基线导入和配置。

## 配置位置

主配置：`D:\finance\model\StockMixer+llm\StockMixer\configs\stockmixer.yaml`。

新闻配置：`D:\finance\model\StockMixer+llm\StockMixer\configs\llm.yaml`。

```yaml
# stockmixer.yaml
market_name: NASDAQ
llm:
  enabled: true
  config: llm.yaml
data:
  protocol: original
```

```yaml
# llm.yaml（节选，不能用本节选覆盖完整文件）
fusion:
  enabled_channels: [market]
  hidden_dim: 8
  dropout: 0.1
  freeze_backbone: true
training:
  epochs: 30
  learning_rate: 0.0001
pipeline:
  datasets: [NASDAQ]
  data_protocol: original
  state: outputs/artifacts/news/nasdaq_market_original_state.json
```

原始 E0 参数仍为窗口16、步长1、alpha=0.1、学习率0.001、100轮和原种子。hidden_dim、dropout、30轮及0.0001仅用于新增新闻分支，是预先固定的初始配置，不保证最优。

本次复用已完成的原始口径 E0：

`outputs/20261116_132645_NASDAQ_E0/checkpoints/best_model.pt`

导入时检查完整非 smoke 训练、数据与股票轴、代码兼容、协议及训练参数。不会为节省时间混用 strict 权重。旧 E0 文件不作修改。

## 准备与检查（不训练）

```powershell
Set-Location 'D:\finance\model\StockMixer+llm\StockMixer'
$smPython = 'C:\Users\marti\anaconda3\envs\stockmixer\python.exe'

# 只读检查，默认行为也是--check
& $smPython -B scripts/15_prepare_market_experiment.py --check

# 复用既有审核、FinBERT向量和基线，生成/检查新关联及面板
& $smPython -B scripts/15_prepare_market_experiment.py --prepare
```

准备脚本不调用训练入口、不运行 Qwen、不加载 FinBERT 做推理。所需既有产物缺失就停止，不自动补训练或推理。原始 JSON 不变。

新闻配置变更后，旧状态不能继续混用；需要新准备版本：

```powershell
& $smPython -B scripts/15_prepare_market_experiment.py --prepare --new
```

准备完成后 `--check` 应显示 `ready: true`、`training_started: false`、`channels: [market]`、`data_protocol: original`。

## 由用户决定何时开始训练和预测

仅当准备检查通过，且你准备开始实验时运行：

```powershell
& $smPython -B src/train_strict.py
```

它读取两个配置、冻结已绑定的原始 E0 主干，训练 Market 分支，按验证 loss 选权重，并依既有流程输出测试与预测。每轮测试仅用于报告，不能用测试成绩选超参数。

本次没有运行这条训练命令，没有开展短训练或正式训练。

## 数据与输出地址

- 原始新闻：`D:\finance\model\StockMixer+llm\data_llm`，只读。
- 输入快照：`outputs/artifacts/news/adapted/20261117_162900_US_GLOBAL_industry_v2`。
- 复用 FinBERT：`outputs/artifacts/news/pipelines/20261117_194036_NASDAQ_NYSE_SP500_preparation/article_features`。
- 本轮状态：`outputs/artifacts/news/nasdaq_market_original_state.json`，与旧的三数据集状态分开。
- 本轮面板：以本轮状态文件 `artifacts.NASDAQ.panel` 为准。
- 实际训练后：`outputs/YYYYMMDD_HHMMSS_NASDAQ_fusion/`，同一次训练与预测在一个目录。
- 本轮修改、检查及备份记录：`outputs/reports/20261118_100440_NASDAQ_market_original_preparation/`。

## 效果的实际边界

当前可用新闻仍少，且原文证据未补齐。准备通过不代表统计样本充分，也不代表预测效果提高。

Market 特征由多只股票共享；现有分支通过基础收益率门控产生一定个股差异，但没有额外的丰富个股条件。因此它可能主要修正收益率水平，对截面排序、prec@10等指标的改善能力有限。应同时检查 MSE、IC、RIC 和收益率排序，不能只看 loss。

先验证相同 E0 下有没有可重复增益，再用固定验证策略比较更小分支、不同新闻表示或补充真实新闻；不根据测试结果反复调参。公司/行业通道待身份和历史成员覆盖补齐后再分别加入。

如果改回 strict：同时修改主配置和 `pipeline.data_protocol`，使用同协议 E0 并新建面板准备版本。准备脚本不会自动训练缺失的 strict E0，也不会把 original 权重当 strict 权重使用。
