# StockMixer + LLM 检查清单与审计记录

## P0：最高优先级——数据覆盖率与数据正确性

### 1. 三个数据集、三个新闻通道的新闻覆盖率审计

统计 NASDAQ、NYSE、SP500 在 Company、Industry、Market 三个通道，以及 Train、Val、Test 三个阶段的有效新闻数量和覆盖率，形成 27 组统计结果。

#### 检查结果

**检查时间：** `2026-11-18T10:49:03+08:00`

**结论：** 27组统计已完成；Company/Industry覆盖明显不足，不能据此称三通道数据充分。

统计口径：按实际面板 provenance 的 source_id 在各通道/阶段去重；同一文章的5日延续不重复计篇数。Train/Val/Test采用名义严格标签边界对应的决策日区间，排除没有16日输入的开头及最后一个无下一日标签的日期。当前original训练可能额外使用验证期标签，见第4项。

Market为全市场共享数组，覆盖率是有效决策日比例；Company/Industry覆盖率是股票×决策日比例，不能直接横比。

| 数据集 | 通道 | 阶段 | 独立新闻篇数 | 有效单元/总单元 | 覆盖率 | 有新闻日期/阶段日期 | 触达股票数/股票池 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| NASDAQ | company | Train | 10 | 46/759240 | 0.006059% | 46/740 | 1/1026 |
| NASDAQ | company | Val | 2 | 7/258552 | 0.002707% | 7/252 | 1/1026 |
| NASDAQ | company | Test | 1 | 5/243162 | 0.002056% | 5/237 | 1/1026 |
| NASDAQ | industry | Train | 2 | 35/759240 | 0.004610% | 10/740 | 6/1026 |
| NASDAQ | industry | Val | 0 | 0/258552 | 0.000000% | 0/252 | 0/1026 |
| NASDAQ | industry | Test | 1 | 10/243162 | 0.004112% | 5/237 | 2/1026 |
| NASDAQ | market | Train | 29 | 126/740 | 17.027027% | 126/740 | 1026/1026 |
| NASDAQ | market | Val | 15 | 68/252 | 26.984127% | 68/252 | 1026/1026 |
| NASDAQ | market | Test | 7 | 32/237 | 13.502110% | 32/237 | 1026/1026 |
| NYSE | company | Train | 7 | 34/1285380 | 0.002645% | 34/740 | 2/1737 |
| NYSE | company | Val | 2 | 10/437724 | 0.002285% | 10/252 | 2/1737 |
| NYSE | company | Test | 2 | 10/411669 | 0.002429% | 10/237 | 2/1737 |
| NYSE | industry | Train | 1 | 5/1285380 | 0.000389% | 5/740 | 1/1737 |
| NYSE | industry | Val | 0 | 0/437724 | 0.000000% | 0/252 | 0/1737 |
| NYSE | industry | Test | 0 | 0/411669 | 0.000000% | 0/237 | 0/1737 |
| NYSE | market | Train | 29 | 126/740 | 17.027027% | 126/740 | 1737/1737 |
| NYSE | market | Val | 15 | 68/252 | 26.984127% | 68/252 | 1737/1737 |
| NYSE | market | Test | 7 | 32/237 | 13.502110% | 32/237 | 1737/1737 |
| SP500 | company | Train | 4 | 20/469260 | 0.004262% | 20/990 | 1/474 |
| SP500 | company | Val | 0 | 0/119922 | 0.000000% | 0/253 | 0/474 |
| SP500 | company | Test | 1 | 5/166848 | 0.002997% | 5/352 | 1/474 |
| SP500 | industry | Train | 0 | 0/469260 | 0.000000% | 0/990 | 0/474 |
| SP500 | industry | Val | 0 | 0/119922 | 0.000000% | 0/253 | 0/474 |
| SP500 | industry | Test | 0 | 0/166848 | 0.000000% | 0/352 | 0/474 |
| SP500 | market | Train | 48 | 220/990 | 22.222222% | 220/990 | 474/474 |
| SP500 | market | Val | 14 | 60/253 | 23.715415% | 60/253 | 474/474 |
| SP500 | market | Test | 15 | 68/352 | 19.318182% | 68/352 | 474/474 |

面板来源（NASDAQ使用当前已绑定版本；NYSE/SP500使用最近已生成三数据集版本，后两者当前未启用训练）：

- **NASDAQ：** [面板目录](D:/finance/model/StockMixer+llm/StockMixer/outputs/artifacts/news/pipelines/20261118_102410_NASDAQ_preparation/NASDAQ_panel)

- **NYSE：** [面板目录](D:/finance/model/StockMixer+llm/StockMixer/outputs/artifacts/news/pipelines/20261117_194036_NASDAQ_NYSE_SP500_preparation/NYSE_panel)

- **SP500：** [面板目录](D:/finance/model/StockMixer+llm/StockMixer/outputs/artifacts/news/pipelines/20261117_194036_NASDAQ_NYSE_SP500_preparation/SP500_panel)

**结构化证据：** [item1.json](audit_20261118_items1-5/item1.json)

### 2. 解决新闻覆盖率不足的问题

根据审计结果，补充历史新闻数据、公司实体映射、历史行业成员关系和 Market Router，提升有效新闻覆盖率。

#### 检查结果

**检查时间：** `2026-11-18T10:50:08+08:00`

**结论：** 第2项尚未解决。已有字典、有限历史主数据及路由产物，但不等于历史新闻与历史成员覆盖充分；本次仅核实，不启动采集、Qwen、FinBERT或训练。

真实FinBERT缓存：1903篇，成功1903篇；有向量不代表能落入目标年份并匹配股票。

NASDAQ：股票池1026只，历史身份有证据股票9只、历史行业有证据股票9只（仅观察日期，不是连续历史）；行情日期2013-01-02～2017-12-08。

面板构建排除计数：

```json
{
  "available_after_dataset": 773,
  "semantic_quarantine": 2,
  "publication_before_dataset_start": 212,
  "unmapped_article": 849,
  "invalid_or_unavailable_membership": 165
}
```

；其中membership等计数可为文章×日期×成员，不能当去重新闻篇数相加。

市场路由决策：

```json
{
  "exclude": 1212,
  "pending": 571,
  "include": 120
}
```

；pending未发布，include也不等于都处于行情日期内。

历史关系未匹配原因：

```json
{
  "unverified_or_unmatched_identity": 4129,
  "semantic_quarantine": 2
}
```

（实体候选级计数）。

NYSE：股票池1737只，历史身份有证据股票3只、历史行业有证据股票3只（仅观察日期，不是连续历史）；行情日期2013-01-02～2017-12-08。

面板构建排除计数：

```json
{
  "available_after_dataset": 773,
  "semantic_quarantine": 2,
  "publication_before_dataset_start": 212,
  "unmapped_article": 851,
  "invalid_or_unavailable_membership": 20
}
```

；其中membership等计数可为文章×日期×成员，不能当去重新闻篇数相加。

市场路由决策：

```json
{
  "exclude": 1212,
  "pending": 571,
  "include": 120
}
```

；pending未发布，include也不等于都处于行情日期内。

历史关系未匹配原因：

```json
{
  "unverified_or_unmatched_identity": 4127,
  "semantic_quarantine": 2
}
```

（实体候选级计数）。

SP500：股票池474只，历史身份有证据股票2只、历史行业有证据股票2只（仅观察日期，不是连续历史）；行情日期2016-01-04～2022-05-25。

面板构建排除计数：

```json
{
  "semantic_quarantine": 2,
  "publication_before_dataset_start": 756,
  "available_after_dataset": 1,
  "unmapped_article": 1060,
  "invalid_or_unavailable_membership": 15
}
```

；其中membership等计数可为文章×日期×成员，不能当去重新闻篇数相加。

市场路由决策：

```json
{
  "exclude": 1212,
  "pending": 571,
  "include": 120
}
```

；pending未发布，include也不等于都处于行情日期内。

历史关系未匹配原因：

```json
{
  "unverified_or_unmatched_identity": 4137,
  "semantic_quarantine": 2
}
```

（实体候选级计数）。

已具备：统一SEC-SIC行业字典、数据集股票轴、证据约束映射及待匹配清单；仍缺：目标历史年份的更多可靠原文、更多股票的历史身份/行业有效期、待审路由的逐条证据。

补齐顺序：先按各数据集真实训练/验证/测试年份补采原文并保留发布时间与采集时间；再补有来源的历史身份/行业区间；复核pending市场/行业作用域；生成新快照与新面板后重做本27组统计。

禁止为增加条数而放宽实体核验、把今日行业回填历史、延长单日SEC观察有效期，或把同一新闻多日聚合当新增新闻。本次没有实际补采，故不能标记覆盖率已提升。

**结构化证据：** [item2.json](audit_20261118_items1-5/item2.json)

### 3. 完成 Company / Industry / Market 三通道映射验证

确保公司新闻对应正确股票、行业新闻对应历史行业成分股、市场新闻对应正确的目标股票池。

#### 检查结果

**检查时间：** `2026-11-18T10:51:41+08:00`

**结论：** 以下结果验证已发布映射的结构、冻结证据和可重放性；不等于全部新闻完成了人工语义审核，也不代表覆盖充分。

NASDAQ：通过。冻结源文件哈希、数据集/股票轴、公司历史身份与字面提及、SEC-SIC字典代码/版本、行业成员关系重新解析、市场路由审核绑定及实际传播对象均已复核。Company链接17条，Industry链接29条，历史成员30条，已发布Market路由120条；以上为链接/路由条数，并非第1项的有效去重新闻篇数。

市场范围：

```json
{
  "legacy_reviewed_US_whole_pool": 120
}
```

，布局shared_date。现有路由是旧版已审核US全市场路由，兼容规则按US_EQUITY全池处理；不是新scope_claims字段已全部补齐。若有NASDAQ_LISTED/NYSE_LISTED必须按历史上市成员逐股传播，不能直接全池广播。

NYSE：通过。冻结源文件哈希、数据集/股票轴、公司历史身份与字面提及、SEC-SIC字典代码/版本、行业成员关系重新解析、市场路由审核绑定及实际传播对象均已复核。Company链接19条，Industry链接29条，历史成员22条，已发布Market路由120条；以上为链接/路由条数，并非第1项的有效去重新闻篇数。

市场范围：

```json
{
  "legacy_reviewed_US_whole_pool": 120
}
```

，布局shared_date。现有路由是旧版已审核US全市场路由，兼容规则按US_EQUITY全池处理；不是新scope_claims字段已全部补齐。若有NASDAQ_LISTED/NYSE_LISTED必须按历史上市成员逐股传播，不能直接全池广播。

SP500：通过。冻结源文件哈希、数据集/股票轴、公司历史身份与字面提及、SEC-SIC字典代码/版本、行业成员关系重新解析、市场路由审核绑定及实际传播对象均已复核。Company链接9条，Industry链接29条，历史成员10条，已发布Market路由120条；以上为链接/路由条数，并非第1项的有效去重新闻篇数。

市场范围：

```json
{
  "legacy_reviewed_US_whole_pool": 120
}
```

，布局shared_date。现有路由是旧版已审核US全市场路由，兼容规则按US_EQUITY全池处理；不是新scope_claims字段已全部补齐。若有NASDAQ_LISTED/NYSE_LISTED必须按历史上市成员逐股传播，不能直接全池广播。

限制：当前主要依据存储摘要及有限SEC历史观察。自动审核、证据重放通过不等于金融语义准确率100%；未匹配实体、未验证行业及pending市场路由继续排除。

映射源与证据均在第1项各面板目录的 relation_*、relations_manifest.json、market_route_*、market_routing_manifest.json 内冻结；具体代码/来源哈希见本项JSON证据。

**结构化证据：** [item3.json](audit_20261118_items1-5/item3.json)

### 4. 完成 Strict 时间防泄露审计

验证新闻发布时间、交易日对齐、决策 cutoff、PIT 行业关系、新闻有效窗口和训练期 PCA，确保不存在未来信息泄露。

#### 检查结果

**检查时间：** `2026-11-18T10:53:17+08:00`

**结论：** 严格PIT无未来信息泄露验收“不通过”。当前为original + retrospective探索性回溯实验；以下分别报告回溯内部时序检查与严格PIT缺口，不能混为通过。

NASDAQ：回溯内部检查通过；核验1245个交易日cutoff（含11个提前收盘日）、365条聚合溯源；未发现的违规项计数为空才算通过：

```json
{}
```

。

发布时间/仅日期顺延、严格早于cutoff、5交易日新闻有效窗口、聚合权重、事件日期历史关系已逐条检查。行业关系使用event_date：核验新闻发生日成员资格，并非每个后续决策日仍具成员资格。

训练期PCA：拟合文章41篇，拟合ID哈希/边界、均值、16维基向量从训练集合重算核对：

```json
{
  "fit_ids_count": 41,
  "manifest_count": 41,
  "fit_source_hash_matches": true,
  "training_boundaries_match": true,
  "mean_matches_train_recompute": true,
  "basis_matches_train_recompute": true
}
```

。跨阶段共享source_id：{"Train/Val": 0, "Train/Test": 0, "Val/Test": 0}；共享本身需按时间窗口解释，不能一概当泄露。

严格observed_pit重判已有有效文章：

```json
{
  "unobserved_availability": 49,
  "after_dataset": 17
}
```

；公司/行业中按known_from不可用于历史决策的溯源贡献110条。2026年采集/推断和证据获知时间不能回填到历史，故严格PIT不通过。

NYSE：回溯内部检查通过；核验1245个交易日cutoff（含11个提前收盘日）、315条聚合溯源；未发现的违规项计数为空才算通过：

```json
{}
```

。

发布时间/仅日期顺延、严格早于cutoff、5交易日新闻有效窗口、聚合权重、事件日期历史关系已逐条检查。行业关系使用event_date：核验新闻发生日成员资格，并非每个后续决策日仍具成员资格。

训练期PCA：拟合文章37篇，拟合ID哈希/边界、均值、16维基向量从训练集合重算核对：

```json
{
  "fit_ids_count": 37,
  "manifest_count": 37,
  "fit_source_hash_matches": true,
  "training_boundaries_match": true,
  "mean_matches_train_recompute": true,
  "basis_matches_train_recompute": true
}
```

。跨阶段共享source_id：{"Train/Val": 0, "Train/Test": 0, "Val/Test": 0}；共享本身需按时间窗口解释，不能一概当泄露。

严格observed_pit重判已有有效文章：

```json
{
  "unobserved_availability": 50,
  "after_dataset": 13
}
```

；公司/行业中按known_from不可用于历史决策的溯源贡献60条。2026年采集/推断和证据获知时间不能回填到历史，故严格PIT不通过。

SP500：回溯内部检查通过；核验1611个交易日cutoff（含12个提前收盘日）、410条聚合溯源；未发现的违规项计数为空才算通过：

```json
{}
```

。

发布时间/仅日期顺延、严格早于cutoff、5交易日新闻有效窗口、聚合权重、事件日期历史关系已逐条检查。行业关系使用event_date：核验新闻发生日成员资格，并非每个后续决策日仍具成员资格。

训练期PCA：拟合文章52篇，拟合ID哈希/边界、均值、16维基向量从训练集合重算核对：

```json
{
  "fit_ids_count": 52,
  "manifest_count": 52,
  "fit_source_hash_matches": true,
  "training_boundaries_match": true,
  "mean_matches_train_recompute": true,
  "basis_matches_train_recompute": true
}
```

。跨阶段共享source_id：{"Train/Val": 0, "Train/Test": 0, "Val/Test": 0}；共享本身需按时间窗口解释，不能一概当泄露。

严格observed_pit重判已有有效文章：

```json
{
  "unobserved_availability": 73,
  "after_dataset": 9
}
```

；公司/行业中按known_from不可用于历史决策的溯源贡献25条。2026年采集/推断和证据获知时间不能回填到历史，故严格PIT不通过。

当前主配置data.protocol=original，新闻pipeline.data_protocol=original；当前NASDAQ original候选目标16～771，验证目标756～1007，候选中重叠756～771共16个验证标签（每轮随机取740个，是否全部采到取决于该轮）。这不是新闻发布日期违规，而是用户选择保留的原始实现训练切分。

original各市场潜在重叠：

```json
{
  "NASDAQ": {
    "target_indices": [
      756,
      771
    ],
    "target_dates": [
      "2016-01-04",
      "2016-01-26"
    ]
  },
  "NYSE": {
    "target_indices": [
      756,
      771
    ],
    "target_dates": [
      "2016-01-04",
      "2016-01-26"
    ]
  },
  "SP500": {
    "target_indices": [
      1006,
      1021
    ],
    "target_dates": [
      "2020-01-02",
      "2020-01-24"
    ]
  }
}
```

整改条件：若验收要求strict标签隔离，需切换两处protocol并使用同协议重训E0；若要求observed PIT，还需有历史真实可用时间及当时已知主数据。仅把mode或时间戳改早不能补齐证据。保留当前探索实验配置，本次不擅自切换或重训。

**结构化证据：** [item4.json](audit_20261118_items1-5/item4.json)

### 5. 完成三个数据集的新闻面板质量检查

确保 NASDAQ、NYSE、SP500 的新闻面板与行情数据的股票轴、日期轴、Mask 和数据切分完全对应。

#### 检查结果

**检查时间：** `2026-11-18T10:54:52+08:00`

**结论：** 结构与数组一致性检查见下；结构通过不等于三通道新闻充分，也不代表严格PIT通过。

NASDAQ：结构检查通过。行情shape=(1026, 1245, 5)；新闻宽度24（PCA16+情绪3+统计5）；股票UID顺序、原始身份数组行号、全部交易日期与两个切分边界均一致。

新闻数组/Mask：

```json
{
  "company": {
    "shape": [
      1026,
      1245,
      24
    ],
    "mask_shape": [
      1026,
      1245
    ],
    "active_cells": 58,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  },
  "industry": {
    "shape": [
      1026,
      1245,
      24
    ],
    "mask_shape": [
      1026,
      1245
    ],
    "active_cells": 45,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  },
  "market": {
    "shape": [
      1245,
      24
    ],
    "mask_shape": [
      1245
    ],
    "active_cells": 226,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  }
}
```

。已核验所有面板manifest文件哈希、冻结映射、数组有限值、二值Mask、缺失位置严格为0、有效情绪概率及provenance与Mask逐单元一致。

名义标签日期：

```json
{
  "Train": [
    "2013-01-25",
    "2015-12-31"
  ],
  "Val": [
    "2016-01-04",
    "2016-12-30"
  ],
  "Test": [
    "2017-01-03",
    "2017-12-08"
  ]
}
```

；original训练额外候选重叠详见第4项。

NYSE：结构检查通过。行情shape=(1737, 1245, 5)；新闻宽度24（PCA16+情绪3+统计5）；股票UID顺序、原始身份数组行号、全部交易日期与两个切分边界均一致。

新闻数组/Mask：

```json
{
  "company": {
    "shape": [
      1737,
      1245,
      24
    ],
    "mask_shape": [
      1737,
      1245
    ],
    "active_cells": 55,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  },
  "industry": {
    "shape": [
      1737,
      1245,
      24
    ],
    "mask_shape": [
      1737,
      1245
    ],
    "active_cells": 5,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  },
  "market": {
    "shape": [
      1245,
      24
    ],
    "mask_shape": [
      1245
    ],
    "active_cells": 226,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  }
}
```

。已核验所有面板manifest文件哈希、冻结映射、数组有限值、二值Mask、缺失位置严格为0、有效情绪概率及provenance与Mask逐单元一致。

名义标签日期：

```json
{
  "Train": [
    "2013-01-25",
    "2015-12-31"
  ],
  "Val": [
    "2016-01-04",
    "2016-12-30"
  ],
  "Test": [
    "2017-01-03",
    "2017-12-08"
  ]
}
```

；original训练额外候选重叠详见第4项。

SP500：结构检查通过。行情shape=(474, 1611, 5)；新闻宽度24（PCA16+情绪3+统计5）；股票UID顺序、原始身份数组行号、全部交易日期与两个切分边界均一致。

新闻数组/Mask：

```json
{
  "company": {
    "shape": [
      474,
      1611,
      24
    ],
    "mask_shape": [
      474,
      1611
    ],
    "active_cells": 25,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  },
  "industry": {
    "shape": [
      474,
      1611,
      24
    ],
    "mask_shape": [
      474,
      1611
    ],
    "active_cells": 0
  },
  "market": {
    "shape": [
      1611,
      24
    ],
    "mask_shape": [
      1611
    ],
    "active_cells": 348,
    "ledger_cells_match": true,
    "dtype": "float32",
    "mask_dtype": "uint8"
  }
}
```

。已核验所有面板manifest文件哈希、冻结映射、数组有限值、二值Mask、缺失位置严格为0、有效情绪概率及provenance与Mask逐单元一致。

名义标签日期：

```json
{
  "Train": [
    "2016-01-27",
    "2019-12-31"
  ],
  "Val": [
    "2020-01-02",
    "2020-12-31"
  ],
  "Test": [
    "2021-01-04",
    "2022-05-25"
  ]
}
```

；original训练额外候选重叠详见第4项。

Mask口径：新闻Mask表示新闻特征可用，行情Mask表示行情/标签可用，两者语义不同，不应强行数值相等；其股票/日期索引必须一致，损失仍按原行情有效性Mask处理。

可用性限制：SP500 Industry三个阶段均为0，NYSE Industry验证和测试为0，NASDAQ Industry验证为0，故三数据集三通道完整实验仍未具备充分数据条件。当前只启用NASDAQ Market，不能直接将当前配置改market_name后宣称其他两市场准备就绪。

配置一致性提示：第7项原文字段“40维新闻特征”与当前实际24维新闻面板不一致；本次按请求保留第6～9项原文，未修改其标题。原行情主干temporal_dim=40与新闻输入维度不是同一参数。

时间限制补充：原始行情manifest已注明保留原始尺度/归一化可能使用全期间信息；本次保留用户选择的original，不将其标记为strict无泄露。新闻数据的结构通过不能消除此限制。

本次只改检查日志与证据文件，未改模型/配置/数据、未发起训练或预测、未操作上游采集器。

**结构化证据：** [item5.json](audit_20261118_items1-5/item5.json)

## P1：高优先级——模型跑通与正式训练

### 6. 跑通原始 StockMixer 训练流程

完成 Train、Val、Test、Checkpoint 保存和预测，建立原始模型基线。

### 7. 跑通新闻融合 StockMixer 训练流程

保持当前 Company、Industry、Market 三通道结构、40 维新闻特征、个股 Gate 和动态衰减机制。

### 8. 完成新闻通道消融实验

分别测试 Company、Industry、Market，以及三通道组合的预测效果。

### 9. 完成 NASDAQ、NYSE、SP500 三个数据集的正式实验

比较 IC、RankIC、收益相关指标及不同年份的稳定性，判断新闻融合是否真正提升预测能力。
