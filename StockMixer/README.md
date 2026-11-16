# StockMixer + LLM 适配版

唯一运行入口仍为 `src/train_strict.py`。活动配置目录只保留两个文件：

| 文件 | 职责 |
|---|---|
| [configs/stockmixer.yaml](configs/stockmixer.yaml) | 原始StockMixer参数、市场切换、数据协议、训练、预测、输出，以及LLM总开关 |
| [configs/llm.yaml](configs/llm.yaml) | 离线FinBERT、新闻面板、市场信息融合结构、正式E0路径及融合分支训练参数 |

## 直接运行

```powershell
& 'C:/Users/marti/anaconda3/envs/stockmixer/python.exe' 'D:/finance/model/StockMixer+llm/StockMixer/src/train_strict.py'
```

无需命令行指定data、output、train或predict。默认读取stockmixer.yaml，训练完成后加载最佳权重预测。相对数据路径以项目根目录为准；llm.config路径以主配置所在目录为准。

## 原始StockMixer模式

```yaml
market_name: NYSE
llm:
  enabled: false
  config: llm.yaml
data:
  protocol: original
```

默认100轮、lr=0.001、alpha=0.1、原始随机种子。主干和get_loss、load_EOD_data、原始评估器保留官方实现。原始get_batch、validate、两轮训练循环已直接从官方train.py执行作独立参考，验证新流程权重、最佳指标及采样顺序一致。

原始模式使用原始pkl/NPY的尺度和dtype，不额外做训练期重标定；SP500沿用原始915起始截断及标签公式。每轮持续打乱同一个arange(valid_index)，取前valid_index-lookback_length-steps+1个；每轮验证与测试，最低验证loss记录best及该轮test。

**原始模式保留官方窗口问题**：NYSE/NASDAQ的候选起点0..755，训练目标可能到771，进入验证期；SP500候选起点0..1005，训练目标可能到1021。该模式是原代码复现，不能称为无泄漏评估。新旧协议结果不要混比。

若需要此前合法目标窗口和训练期缩放，将data.protocol改为strict即可；prepared_root保持null时会自动选择独立数据目录。主干、损失和五项官方评估定义不变。

## 启用LLM市场信息融合

1. 在llm.yaml填写baseline_checkpoint和所选市场的panels路径，必须是真实且通过核验的同协议产物。
2. 在stockmixer.yaml将llm.enabled改为true。
3. 运行同一个src/train_strict.py。

关闭开关时，不读取llm.yaml、不加载新闻面板、不创建新闻分支。开启时，读取llm.yaml的融合结构及training参数，冻结该正式E0主干，只训练新Adapter/Gate。主配置仍控制市场、原损失alpha、种子、设备和输出。

当前面板及正式原始协议E0路径尚未填写；开启但缺失这些真实产物会在训练前报错，不会生成虚假新闻或静默降级。旧strict E0不能用于original模式融合，数据哈希和协议会拒绝混用。

Qwen/FinBERT在离线阶段运行，训练forward及epoch中不执行NLP模型。新闻原始JSON只读，缺失通道修正严格为0。默认仅启用market通道，时间模式为retrospective。

## 结果位置

训练目录统一命名为outputs/YYYYMMDD_HHMMSS_数据集_E0或fusion，训练与自动预测保存在同一实验目录，预测CSV位于predictions/predictions.csv，预测元数据位于tables/prediction_metrics.json；工作流报告位于reports下并加_workflow；自定义输出名称也自动补齐前缀。目录中仍有checkpoints、configs、figures、logs、predictions、tables六类目录。工作流报告保存配置原文、解析配置和所有实际读取配置的SHA256。

original行情产物：outputs/artifacts/market_original/{NASDAQ,NYSE,SP500}；strict行情产物继续保留在outputs/artifacts/market。tables/epoch_samples.json记录original模式每轮真实采样顺序。检查tables/metrics.json及predictions/predictions.csv；latest代表历史数据末日，不是实时行情。

旧配置原字节归档到outputs/archive/configs_before_split_20261116；历史checkpoint及旧schema配置继续可读取，不改写旧产物。完整对齐清单与验收放在outputs/reports/two_configs_20261116。

本项目为StockMixer适配修改版，主干仍为released 16+16+8=40。原始代码及论文参考D:/finance/baseline/StockMixer-master；原论文结果参考项目原始stockmixer结果.png。
