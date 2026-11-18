# StockMixer + LLM 适配版

当前主干保留原始StockMixer；公司、行业、市场三个新闻通道在离线准备后参与融合。原始代码及论文位于`D:/finance/baseline/StockMixer-master`。

**Retrace完成后的使用说明：[QUICKSTART.md](docs/QUICKSTART.md)。**

```powershell
conda activate stockmixer
cd 'D:\finance\model\StockMixer+llm\StockMixer'
python main.py
```

`main.py`根据两个config自动完成全量retrace核验/导入、FinBERT编码、映射与面板、最终验收、同口径E0、融合训练、验证/测试，以及最佳权重预测。新闻或覆盖验收未通过时停止，保留错误和报告。

| 配置 | 内容 |
|---|---|
| [stockmixer.yaml](configs/stockmixer.yaml) | 市场、LLM总开关、原始主干与E0训练参数、设备、预测 |
| [llm.yaml](configs/llm.yaml) | retrace路径、目标数据集、FinBERT、三个通道、关系映射、时间协议、门槛及融合训练参数 |

分阶段运行：

```powershell
python scripts/18_import_news.py
python scripts/19_prepare_news.py
python scripts/20_train_test.py
```

18只导入；19完成新闻准备与验收，并核验复用或训练E0；20训练融合分支、测试并按开关预测。`python main.py --step status`只读查看状态，`--step check`只核验全量源数据和配置。重复运行会复核并复用已完成阶段；重新实验用`python main.py --new`。

`src/train_strict.py`继续作为已准备数据的单市场训练/预测入口。协议由config决定，文件名不强制strict。纯StockMixer将`llm.enabled`关闭后运行main.py或train_strict.py。

当前配置为NASDAQ、原始`original_reproduction`与`retrospective`探索性实验，启用市场和行业通道。原始训练候选存在验证标签重叠；无日期行业参考成员也不等同于真实历史成员，不得称为无泄漏PIT。切换strict须准备同协议数据及E0。

训练、测试与自动预测统一保存在`outputs/YYYYMMDD_HHMMSS_数据集_E0或fusion/`；新闻准备及完整验收位于`outputs/artifacts/news/`。详细检查口径见[最终验收说明](docs/FINAL_NEWS_ACCEPTANCE.md)。
