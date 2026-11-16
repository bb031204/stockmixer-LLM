# 统一 outputs 输出目录

所有项目生成的训练、预测、审计及离线新闻产物统一位于：

`D:/finance/model/StockMixer+llm/StockMixer/outputs`

训练与预测采用原StockMixer的时间戳命名和分类目录：

```text
outputs/
  20261113_133214_NYSE_E0/
    checkpoints/       best_model.pt、last_model.pt
    configs/           config.yaml、config.json、model_summary.txt、data_contract.json
    figures/           loss_curve.png、ic_curve.png
    logs/              train.log、predict.log、run_status.json
    predictions/       test_predictions.npz、predictions.csv
    tables/            history.csv、history.json、metrics.csv、metrics.json、sample_ledger.json
    _artifact/         不可变原始文件与原始manifest
    _prediction_artifact/  本次自动预测的不可变证据
    manifest.json      训练与预测文件及不可变产物之间的校验关系
  artifacts/
    market/            NASDAQ、NYSE、SP500严格行情产物的入口
    news/adapted/      父快照、固定1423篇及8篇排除清单
    news/article_features/
    news/cache/
    news/panels/
    schedules/
  reports/             审计、测试与迁移验收
  archive/runs/        已有历史产物的逐字节归档
  README.md            当前实验索引
```

新实验目录统一为 `YYYYMMDD_HHMMSS_数据集_其他信息`，日期时间按 Asia/Taipei（UTC+8）记录。例如 `20261116_093015_NASDAQ_E0`、`20261116_093015_NASDAQ_fusion`；smoke 验证附加 `_smoke`。同次训练和自动预测共用一个实验目录，不再创建同级 `_prediction` 目录。预测CSV保存在 `predictions/predictions.csv`，预测元数据保存在 `tables/prediction_metrics.json`，日志保存在 `logs/predict.log`。训练与预测的原始证据分别位于 `_artifact` 和 `_prediction_artifact`，统一manifest进行校验。工作流报告仍位于共享 `reports/` 下并加 `_workflow`。默认目录同秒重名增加序号；显式目录拒绝覆盖。稳定的行情、新闻缓存路径保留原位；已有成对预测结果的整理见下文。

主配置 `output.directory: null` 自动命名；若填写 `outputs/my_run`，自动生成 `outputs/YYYYMMDD_HHMMSS_NASDAQ_E0_my_run`（按当前数据集及模式替换）。已符合格式的完整名称予以保留，但名称中的数据集必须与配置一致。

## 运行方式

```powershell
& 'C:/Users/marti/anaconda3/envs/stockmixer/python.exe' 'D:/finance/model/StockMixer+llm/StockMixer/src/train_strict.py'
```

只需运行一个文件。configs/stockmixer.yaml控制数据集、训练、设备、输出和预测。训练结果保存在outputs/时间戳_数据集_E0；自动预测写入同一实验目录的predictions/predictions.csv，不替换训练权重、配置或训练指标。全流程报告保存在outputs/reports/训练目录名_workflow，包含配置原文、总日志及训练/预测状态。

下述CLI路径规则适用于内部开发工具；日常训练无需传这些参数。

CLI显式的 `--output outputs/xxx` 或 `--output xxx` 都按项目根目录解析；不受启动PowerShell目录影响。旧的相对 `--output runs/xxx` 会提示并转写到 `outputs/xxx`。绝对输出目录也必须在项目outputs内，不会把生成结果写入原始数据目录。

新闻适配、编码、交易日历、面板和快照命令的output可省略，分别自动归入artifacts下对应分类；编码cache默认在outputs/artifacts/news/cache。原始行情dataset和data_llm/US、GLOBAL继续作为只读输入保留原位。

## 迁移与兼容

原runs和data_llm/adapted已整体移入outputs，文件哈希逐项核对。旧路径仅保留Windows目录联接，实际文件均在outputs，不是两份独立数据。这样既有checkpoint中保存的数据路径、已发布报告和父证据引用仍可用。

迁移后的训练目录附带 `_artifact`，保持原checkpoint及manifest字节不变；对外使用checkpoints下的权重，加载时同时验证展示文件、原始产物和原数据哈希。不得手动编辑 `_artifact`。已审查的阶段1/正式E0代码指纹保持兼容，未知代码版本仍会拒绝。

旧训练/预测包装入口已移除，当前使用src/train_strict.py自动完成全流程；主干、损失函数和合法采样保持不变，指标和逐轮评估已按原始StockMixer调整。默认每轮验证后测试，生成Train/Valid/Test loss、验证/测试IC、RIC和SR曲线。历史只测试一次的实验继续保留原记录，不补造逐轮test。

## 两个配置与原始协议

默认读取stockmixer.yaml；仅llm.enabled=true时再读取llm.yaml。默认original协议的行情产物另存outputs/artifacts/market_original，保留原始尺度及可能进入验证期的原始采样；strict产物仍位于market。每轮真实采样另存tables/epoch_samples.json，完整来源配置见reports下的configuration_sources.json及原文快照。旧三个配置原字节归档到outputs/archive/configs_before_split_20261116。


## 历史结果整理

NASDAQ、NYSE、SP500的三组成对结果已整理为同一个训练目录，预测CSV与预测元数据置于新格式的标准路径。旧预测完整bundle移入outputs/archive/20261116_213325_NASDAQ_NYSE_SP500_layout_migration，以保留历史日志和审计引用的证据。工作流路径及outputs/README.md索引已更新，权重、训练指标和CSV未重新计算。迁移证据见outputs/reports/20261116_213325_NASDAQ_NYSE_SP500_layout_migration。
