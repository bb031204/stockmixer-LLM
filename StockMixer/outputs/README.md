# StockMixer 实验结果

每次训练与自动预测保存在同一个实验目录。预测CSV为 `predictions/predictions.csv`，训练指标为 `tables/metrics.json`，预测元数据为 `tables/prediction_metrics.json`。

| 实验目录 | 数据集 | 类型 | 预测行数 |
|---|---|---|---:|
| [20261116_132645_NASDAQ_E0](20261116_132645_NASDAQ_E0/) | NASDAQ | 正式实验 | 1026 |
| [20261116_140926_NYSE_E0](20261116_140926_NYSE_E0/) | NYSE | 正式实验 | 1737 |
| [20261116_141007_SP500_E0](20261116_141007_SP500_E0/) | SP500 | 正式实验 | 474 |
| [20261116_150000_NASDAQ_E0_single_folder_smoke](20261116_150000_NASDAQ_E0_single_folder_smoke/) | NASDAQ | smoke验证 | 1026 |

行情、新闻缓存位于 `artifacts/`，工作流和核验报告位于 `reports/`。

旧的分离预测目录已逐字节归档到 [历史预测证据](archive/20261116_213325_NASDAQ_NYSE_SP500_layout_migration/)。训练权重、原始训练证据、历史指标和预测CSV字节未改变。迁移前的根manifest和工作流记录保存在 [迁移备份](reports/20261116_213325_NASDAQ_NYSE_SP500_layout_migration/backup/)。

`_artifact`、`_prediction_artifact` 和manifest用于完整性验证，请勿手动改写。
