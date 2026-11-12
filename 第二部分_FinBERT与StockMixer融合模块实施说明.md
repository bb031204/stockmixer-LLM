# 第二部分：FinBERT 新闻编码与 StockMixer 三通道融合模块实施说明

> **用途：** 交给 Vibecoding / Coding Agent 的第二阶段开发任务书。请先检查本地仓库，再按本文要求增量开发、测试和汇报。  
> **目标仓库：** `https://github.com/bb031204/stockmixer-LLM`  
> **本地新闻数据目录：** `D:\finance\model\StockMixer+llm\data_llm`  
> **本阶段范围：** 在现有基础 StockMixer 上增加第一版 FinBERT 离线文本编码、公司/行业/市场三通道新闻特征、时间对齐、数据面板和轻量融合模型。  
> **核心原则：** 保留原始 StockMixer 作为可复现基线；FinBERT 只做离线特征提取；训练时不调用 FinBERT 或 Qwen；所有新闻特征必须严格按样本预测时点构建；不能把回溯实验冒充严格 Point-in-Time（PIT）实验。

---

## 0. 给 Coding Agent 的执行要求

本文件是实施规格，不是要求重写 StockMixer。请按以下顺序工作：

1. 检查本地仓库的真实目录、Git 状态、当前分支、Python/PyTorch 版本、训练入口、模型类、数据加载器、原始数据文件、股票顺序映射和日期轴。
2. 用户给出的 GitHub 仓库链接在本次研究中未能通过 GitHub 文件接口读取（返回 404）。因此，本文对基础代码的核验以 StockMixer 官方上游仓库为参照；**不得把上游文件名、参数或目录结构直接当成用户本地 fork 的事实**。必须以本地仓库实际内容为准完成映射。
3. 先输出只读盘点报告：实际文件清单、张量 shape、日期范围、股票顺序来源、训练/验证/测试边界、当前预测目标和损失函数。盘点完成前不要改代码或数据。
4. 将本阶段新增代码尽量放在独立模块中，例如 `finbert_encoder.py`、`news_panel_builder.py`、`model_with_news.py`、`train_with_news.py`。实际文件名按仓库风格调整。
5. 原始 StockMixer 模型、训练脚本和评估结果必须保留为无新闻基线。优先新增包装模型和训练入口，不要直接大幅修改原 `model.py`。
6. 所有新特征都必须带 `source_id`、模型版本、Schema 版本、时间可用性和数据集/股票轴信息，能够追溯和重建。
7. 不能猜测股票代码、交易所、股票数组行号、历史行业分类、历史指数成分、文章发布时间或文章的真实可用时间。
8. 不得读取、打印、提交或泄露任何 API 密钥、代理凭证或其他秘密。
9. 不得将原始文章全文复制到本项目。第一部分规定原文处理完成后删除；本阶段只使用正式 JSON 摘要、标题及结构化 Parquet 数据。
10. 先做离线合成数据测试，再做真实数据小样本验证。未经用户确认，不批量覆盖或重写原始行情数据、新闻数据和历史映射数据。

### 本阶段明确不做

- 不重新训练或微调 Qwen3:8B。
- 第一版不微调 FinBERT；FinBERT 默认冻结，仅做推理。
- 不在 StockMixer 的 `forward()` 或每个 epoch 内调用 Hugging Face 在线接口。
- 不把 FinBERT 情绪概率当作股票收益率或涨跌标签。
- 不让 LLM 直接生成 NASDAQ / NYSE / S&P 500 的股票数组或 `array_index`。
- 不用当前股票池、当前行业分类或当前 S&P 500 成分名单回填全部历史日期。
- 不把新闻发布日期直接等同于特征可用时间。
- 不随机拆分新闻文章或交易日来训练/验证/测试。
- 不为了增加新闻覆盖率而把所有行业新闻扩散给所有股票。
- 不改变原始 StockMixer 的标签定义、损失函数和行情数据口径，除非单独设计并记录实验版本。

---

## 1. 基础 StockMixer：已核验的上游结构与本地核验要求

### 1.1 官方上游代码结构

StockMixer 官方仓库说明其模型由 indicator mixing、temporal mixing 和 stock mixing 构成；官方 README 说明 NASDAQ / NYSE 数据源来自 Temporal Relational Ranking for Stock Prediction，S&P 500 数据源来自 ESTIMATE 数据集，并将预处理数据存为 `.pkl` / `.npy` 以提高读取速度。

上游仓库可作为代码参照：

- `README.md`
- `src/train.py`
- `src/load_data.py`
- `src/model.py`
- `src/evaluator.py`
- `dataset/NASDAQ/`
- `dataset/NYSE/`
- `dataset/SP500/`

官方上游仓库链接：<https://github.com/SJTU-DMTai/StockMixer>

### 1.2 上游代码中已核验的数据接口

官方上游 `train.py` 对 NASDAQ / NYSE 读取以下 pickle 文件：

| 文件 | 上游用途 | 典型 shape |
|---|---|---|
| `eod_data.pkl` | 股票日度行情特征 | `[N, T, F]` |
| `mask_data.pkl` | 股票-日期有效性掩码 | `[N, T]` |
| `gt_data.pkl` | 每只股票每个日期的收益率标签 | `[N, T]` |
| `price_data.pkl` | 基准收盘价 | `[N, T]` |

其中 `N` 是股票数，`T` 是日期数，`F` 是行情特征数。官方示例中 `fea_num=5`，但必须以用户本地数组的真实 shape 为准。

官方 `load_data.py` 从每只股票的 CSV 读取日度记录，保留日期以外的行情列，并将最后一列作为价格相关字段；缺失值使用 `-1234` 哨兵识别并构造 mask。不得把 `-1234` 当成真实价格或真实行情值。

官方上游的 S&P 500 分支读取 `dataset/SP500/SP500.npy`，代码中使用 `data[:, 915:, :]`，并把最后一个特征列作为价格，基于相邻日期计算收益率。**915 是上游脚本中的切片位置，不是可跨版本照搬的固定日期常量。** 必须用日期元数据确认该位置确实对应本地 S&P 500 数据的起始日。

### 1.3 上游模型的预测口径

官方 `StockMixer.forward()` 输出预测价格；官方 `get_loss()` 再通过：

\[
\widehat r_{i,t+1}
=
\frac{\widehat P_{i,t+1}-P_{i,t}}{P_{i,t}}
\]

将预测价格转换为收益率，与 `gt_data` 比较。损失由收益率 MSE 与 pairwise ranking loss 组成：

\[
L = L_{\mathrm{MSE}}+\alpha L_{\mathrm{rank}}
\]

上游 `train.py` 的典型配置包括：

- `lookback_length = 16`
- `steps = 1`
- `alpha = 0.1`
- `learning_rate = 0.001`
- `scale_factor = 3`
- `activation = GELU`
- `market_num` 按数据集设定
- 上游论文报告的 market mixing 超参数：NASDAQ 20、NYSE 25、S&P 500 8

这些是论文/上游参考值，不是用户本地仓库的既定事实。应先读取本地配置；如要复现论文，才按数据集设置对应参数。

### 1.4 必须检查的上游代码风险

上游 `train.py` 示例将 `valid_index=756`、`test_index=1008` 写为全局常量，而论文的 S&P 500 划分为 1006 / 253 / 352 天。因此，**不能让三个数据集共用同一组边界**。本阶段必须建立数据集独立配置，并用实际日期轴核验。

另外，上游训练示例在每个 epoch 都会计算测试集表现。新训练流程应改为：训练期间只根据验证集选模；最佳 checkpoint 确定后，测试集只评估一次，避免反复查看测试集造成研究者层面的测试集过拟合。

---

## 2. NASDAQ、NYSE、S&P 500 数据范围与切分协议

### 2.1 论文公布的数据集统计

StockMixer AAAI 2024 论文 Table 1 公布：

| 数据集 | 股票数 | 开始日期 | 结束日期 | Train Days | Val Days | Test Days |
|---|---:|---|---|---:|---:|---:|
| NASDAQ | 1,026 | 2013-01-02 | 2017-12-08 | 756 | 252 | 273 |
| NYSE | 1,737 | 2013-01-02 | 2017-12-08 | 756 | 252 | 273 |
| S&P 500 | 474 | 2016-01-04 | 2022-05-25 | 1,006 | 253 | 352 |

论文来源：<https://ojs.aaai.org/index.php/AAAI/article/view/28681>  
论文 PDF：<https://ojs.aaai.org/index.php/AAAI/article/download/28681/29322>

以上为论文报告的样本统计。它们不能替代本地数据文件中的真实交易日期列表。实施时必须检查：

- 实际 `T` 是否与数据说明一致。
- 日期轴是否连续、递增、无重复。
- NASDAQ 与 NYSE 是否使用相同日期轴。
- S&P 500 切片后的第一天和最后一天是否与论文报告一致。
- 每只股票的行顺序是否与 `stock_identity_mapping` 一致。
- `eod_data`、`mask_data`、`gt_data`、`price_data` 是否完全对齐。

如果论文公布的天数与本地数组长度不一致，停止训练配置生成，先查明原始数据版本、日期轴、切片和缺失日处理差异。不得静默截断或补日期来凑数量。

### 2.2 数据集独立切分配置

在核实本地日期轴确实与论文版本一致后，以下边界可作为论文复现的初始配置：

| 数据集 | `train_end_idx` | `valid_end_idx` | 测试结束 | 备注 |
|---|---:|---:|---|---|
| NASDAQ | 756 | 1008 | `T` | 756 + 252 + 273 |
| NYSE | 756 | 1008 | `T` | 756 + 252 + 273 |
| SP500 | 1006 | 1259 | `T` | 1006 + 253 + 352 |

配置示意：

```yaml
datasets:
  NASDAQ:
    expected_stocks: 1026
    expected_start: "2013-01-02"
    expected_end: "2017-12-08"
    train_end_idx: 756
    valid_end_idx: 1008
    lookback_length: 16
    steps: 1
    market_num: 20

  NYSE:
    expected_stocks: 1737
    expected_start: "2013-01-02"
    expected_end: "2017-12-08"
    train_end_idx: 756
    valid_end_idx: 1008
    lookback_length: 16
    steps: 1
    market_num: 25

  SP500:
    expected_stocks: 474
    expected_start: "2016-01-04"
    expected_end: "2022-05-25"
    train_end_idx: 1006
    valid_end_idx: 1259
    lookback_length: 16
    steps: 1
    market_num: 8
```

上述 `expected_*` 是核验断言，不是自动补齐目标。若实际数据不符合，必须报告并要求确认正确数据版本。

### 2.3 样本窗口、预测日和标签日的精确关系

对 lookback=16、steps=1 的样本：

- 输入行情窗口：`[s, s+15]`。
- 决策日期索引：`t = s+15`。
- 新闻特征只能使用决策截止时间之前可用的信息。
- 标签日期索引：`t+1 = s+16`。
- `gt_data[:, t+1]` 表示从 `P_t` 到 `P_{t+1}` 的收益率。
- `price_data[:, t]` 是收益率计算的基准价格。

即：

\[
X_{i,t} = [x_{i,t-15}, \ldots, x_{i,t}]
\]

\[
y_{i,t+1} = \frac{P_{i,t+1}}{P_{i,t}}-1
\]

\[
News_{i,t} =
\operatorname{Aggregate}
\{a:\ a \text{ 在决策截止时刻 } C_t \text{ 已可用}\}
\]

**训练/验证/测试应按标签日期索引划分，而不是按新闻文章随机划分。**

以论文边界为例：

- NASDAQ / NYSE：训练标签日期索引位于训练区间，验证标签日期索引为 `[756, 1008)`，测试标签日期索引从 `1008` 开始。
- S&P 500：验证标签日期索引为 `[1006, 1259)`，测试标签日期索引从 `1259` 开始。

每个样本的输入窗口可以自然使用此前日期的数据，包括训练区间末尾之前的历史行情；但任何训练样本的标签都不得落入验证或测试区间。验证/测试样本使用此前 16 日行情作为上下文是允许的，这不是标签泄漏。

---

## 3. 最重要：Point-in-Time 时间对齐与防止数据泄漏

本项目的新闻模块能否得到可信结果，首先取决于时间协议，而不是 FinBERT 的准确率。

### 3.1 必须区分的时间

每篇文章至少保留：

| 字段 | 含义 |
|---|---|
| `published_at` | 来源声称的发布时间 |
| `publication_precision` | `date` / `minute` / `second` |
| `first_seen_at` | 爬虫首次发现文章的真实时间 |
| `fetched_at` | 文章被实际抓取的时间 |
| `feature_available_at` | 第一部分摘要及结构化结果正式可用的时间 |
| `finbert_completed_at` | FinBERT 推理完成的时间 |
| `news_feature_available_at` | 当前新闻特征版本正式生成并可供模型读取的时间 |
| `availability_basis` | `observed` / `verified_audit` / `migration` / `simulated` |
| `model_available_at` | 当前 FinBERT 模型版本可被外部使用的时间 |
| `mapping_known_at` | 当前证券/行业映射被系统实际获知的时间 |

对新闻特征版本，保守可用时间至少为：

\[
A(a)=\max(
published\_at,\ first\_seen\_at,\ fetched\_at,\
feature\_available\_at,\ finbert\_completed\_at
)
\]

实际构建面板时，还要检查映射数据是否在决策时点已知，以及当前模型版本是否在决策时点已经可用。任何必要时间缺失时，不得把文章自动视为 PIT 可用。

### 3.2 必须区分三种实验模式

#### Mode A：`observed_pit`（真实观测 PIT）

只有在以下条件全部满足时，才允许使用：

- 文章真实的首次发现、抓取和特征生成时间可追溯。
- `news_feature_available_at <= decision_cutoff_at`。
- FinBERT 版本在决策时点已经发布并可用。
- 证券/行业/指数成分映射在决策时点已知。
- 使用当时可见的新闻和映射版本构建特征。

这是最严格、最接近真实可部署系统的历史回测。

#### Mode B：`simulated_pit`（模拟延迟回放）

如果历史文章只有可信发布时间，没有当时的抓取/推理时间，可以建立模拟延迟规则，例如：

- 文章在来源发布时间之后才进入系统。
- 推理延迟使用明确配置的固定或经验分布。
- 只有 `published_at + simulated_latency` 不晚于决策截止时间，文章才进入特征。
- 记录 `availability_basis=simulated`，并将整个实验标记为模拟回放。

**模拟延迟不等于真实历史采集记录。** 结果只能描述为“在假设采集/推理延迟下的历史回放”，不得宣称严格真实 PIT。

#### Mode C：`retrospective`（回溯性文本研究）

允许对历史文章使用后来发布的 FinBERT、后来生成的摘要或后来整理的行业映射。可以研究文本特征与历史收益的统计关系，但不能声称这些特征在历史交易日已经可用，也不能把结果称为严格 PIT 可交易回测。

每次实验必须在 manifest、日志和结果表中写清模式，不允许将三种模式混合报告。

### 3.3 FinBERT 发布时间造成的历史限制

本阶段第一版建议使用 `ProsusAI/finbert`。其相关 FinBERT 论文发表于 2019-08-27，模型卡标注其为英文金融情绪模型。NASDAQ / NYSE 数据集结束于 2017-12-08，早于该论文发布时间。

因此：

- NASDAQ / NYSE 全部历史区间不能使用该模型声称严格历史 PIT 可部署。
- S&P 500 的 2016 年至 FinBERT 可用日期之前的部分也不能声称严格 PIT。
- S&P 500 后续区间只有在模型版本发布日、实际新闻特征可用时间和映射已知时间均有证据时，才有资格进入 `observed_pit`。
- 不能因为模型今天可以下载，就假定它在 2013、2016 或 2017 年已经存在。
- 对完整 NASDAQ / NYSE 历史区间，第一版 FinBERT 实验应明确标记为 `retrospective` 或 `simulated_pit`，不能伪装为 `observed_pit`。

模型论文与模型卡：

- <https://arxiv.org/abs/1908.10063>
- <https://huggingface.co/ProsusAI/finbert>

### 3.4 交易所时间与新闻进入日期

NASDAQ、NYSE、S&P 500 均按美国市场交易日处理，使用 `America/New_York` 时区和真实交易日历（包括夏令时、节假日和提前收盘）。

以“在交易日 t 收盘时做预测、预测 t 到 t+1 收盘收益率”为研究定义时：

- 只允许使用在 `t` 的决策截止时刻之前已经可用的新闻特征。
- 盘前发布且已完成抓取/推理的文章可进入当天特征。
- 盘中发布的文章只有在决策截止前已完成处理时才能进入当天特征。
- 收盘后发布的文章进入下一交易日或之后的第一个可用决策时点。
- 只有日期、没有时分秒的文章，不能假定盘前发布；保守地从下一交易日开始可用。
- 发布时间虽早，但抓取或 FinBERT 推理在决策截止之后完成的文章，不得进入该日特征。
- 遇到时区缺失、发布时间歧义或交易日历无法确认的文章，标记为不可用于严格 PIT，不得猜测。

注意：原始 StockMixer 的标签是收盘价到下一交易日收盘价的收益率。若要声称真实可交易策略，还需另外定义订单执行时间、下一开盘入场、交易成本和滑点；不能把收盘到收盘的预测指标自动解释为可实现收益。

### 3.5 防止其他前视泄漏

必须同时防止：

1. **新闻时间泄漏：** 不得使用决策截止时间之后发布或处理完成的文章。
2. **摘要时间泄漏：** 不能把今天才生成的摘要冒充历史当天已可用。
3. **FinBERT 模型泄漏：** 模型版本必须有固定 revision 和可用日期。
4. **股票映射泄漏：** 不能用未来才知道的 ticker 更名、并购、上市/退市关系匹配过去新闻。
5. **行业映射泄漏：** 不能用今天的行业分类无条件回填历史。
6. **指数成分泄漏：** 不能用当前 S&P 500 成分名单回填 2016 年至今。
7. **标准化泄漏：** scaler、PCA、缺失值填补参数只能在训练集拟合。
8. **超参数泄漏：** 时间衰减、阈值、通道权重、模型结构只根据训练/验证选择，不能根据测试集选择。
9. **事件聚类泄漏：** 不能利用未来文章完成全时期聚类后，再用聚类结果修改过去时点的去重或聚合。每个 cutoff 只能看到当时已可用的文章。
10. **测试集反复窥视：** 训练期间不反复报告/选择测试表现；最终 checkpoint 锁定后测试一次。
11. **随机拆分泄漏：** 不得随机拆分新闻文章或交易日；必须按时间顺序切分。
12. **重叠窗口泄漏：** 允许验证/测试样本使用过去行情作为输入上下文，但训练样本标签不能越过训练边界。

### 3.6 历史映射采用双时间（Bitemporal）字段

证券主数据、行业分类和指数成分关系建议同时记录：

- `valid_from` / `valid_to`：该关系在现实世界中何时有效。
- `known_from` / `known_to`：系统何时实际知道或持有该版本关系。

严格 PIT 查询必须同时满足：

\[
valid\_from \le t < valid\_to
\]

\[
known\_from \le decision\_time < known\_to
\]

只有 `valid_from` 而没有 `known_from` 的历史映射，只能标记为历史重建数据，不得声称完全复原了当时的信息集。

---

## 4. 第一部分数据接口：第二部分的输入约定

本阶段消费第一部分爬虫输出，不重新爬取文章、不重新调用 Qwen。

建议从 `data_llm/input/` 读取第一部分正式导出的数据；如果第一部分输出目录不同，通过配置指定，不要在代码中硬编码多个路径。

### 4.1 主要输入表

| 输入文件 | 用途 |
|---|---|
| `article_metadata.parquet` | 文章 ID、标题、市场、语言、时间、可用时间、事件类型 |
| `article_entities.parquet` | 公司/发行人实体、角色、Ticker、交易所、映射状态 |
| `article_industries.parquet` | 行业原文、标准行业代码、explicit / inferred 依据 |
| `article_events.parquet` | 事件类型和关键事实 |
| `article_industry_impacts.parquet` | 行业影响角色、机制、条件性方向 |
| `security_master.parquet` | 证券主数据 |
| `security_industry_history.parquet` | 历史行业分类与有效期 |

实际文件名和字段以第一部分落地后的 Schema 为准。必须读取 Schema 版本并验证，不得静默把缺失字段填成确定事实。

### 4.2 正式 JSON 的兼容读取

如果某些历史文章还没有 Parquet，允许从第一部分正式 JSON 兼容读取：

- `source_id`
- `url`
- `时间`
- `时间信息`
- `标题`（如存在）
- `主要内容`
- `股票实体`
- `涉及行业`
- `影响范围`

只使用标题、摘要和结构化字段。**不允许在本阶段恢复、缓存或保存原始文章全文。**

### 4.3 市场归属和信息层级不能混淆

- `market` / `CN` / `US` / `GLOBAL` 表示新闻数据的市场归属。
- `impact_scope` / `影响范围` 表示 `company` / `industry` / `market` 信息层级。
- 不能因为文章放在 `US` 目录就自动将它标记为 `market-level`。
- 本项目三个股票数据集均为美国股票数据。默认允许 `US` 和 `GLOBAL` 新闻进入美国市场特征；`CN` 新闻默认不直接进入，除非未来建立并验证跨市场传导规则。

---

## 5. 第一版 FinBERT：冻结模型、离线推理

### 5.1 模型选择

第一版固定使用：

```text
Hugging Face model id: ProsusAI/finbert
Task: financial sentiment classification
Language: English
Labels: positive / negative / neutral
Hidden size: 768
Max position length: 512 tokens
```

模型卡说明其为 BERT-base 结构（12 层、768 hidden size、12 attention heads），分类标签顺序为：

- `0`: positive
- `1`: negative
- `2`: neutral

推理输出必须读取模型配置中的 `id2label`，并在保存时统一为固定顺序：

```text
sentiment_probs = [p_positive, p_negative, p_neutral]
```

模型卡：<https://huggingface.co/ProsusAI/finbert>

### 5.2 许可证警告

Hugging Face 当前模型卡标注权重采用 `CC BY-NC-SA 4.0`，存在非商业使用限制。使用前必须检查实际下载 revision 的许可证和用途是否符合要求。训练代码仓库的许可证不等于模型权重许可证。

- 仅用于非商业研究时，也要保留模型归属和许可证信息。
- 如果项目可能用于商业、投资管理、商业化产品或其他不明确用途，先完成许可审查；不能把“开源可下载”理解为可商业使用。
- 不得通过修改模型文件或更换加载方式规避许可证。

### 5.3 输入文本策略

第一部分明确不长期保存原始文章，因此第一版 FinBERT 输入固定为：

```text
主标题 + 正式摘要（主要内容）
```

规则：

1. 优先使用 `标题` 和 `主要内容`。
2. 标题缺失时只用摘要。
3. 摘要缺失或无有效文字时跳过并记录原因。
4. 同一文章标题与摘要重复时先去重。
5. 不使用评论、推荐文章、网页导航、广告或完整 HTML。
6. 不使用未来生成的新闻摘要或未来版本的文章内容。
7. 保存 `input_text_hash`，不保存全文输入。
8. 输入语言必须是英文 `en`，或者由可靠语言识别器判定为英文。
9. 中文和其他非英文文章第一版不送入英文 FinBERT；设置 `finbert_status=unsupported_language`，对应通道 mask 不得假装有有效特征。
10. 第一版不使用翻译模型把中文新闻翻译成英文，避免引入额外模型版本、翻译延迟和语义误差。后续如增加翻译，必须作为独立实验分支。

### 5.4 Tokenization 和推理约束

- 使用本地固定模型目录或 Hugging Face 固定 commit revision。
- 禁止使用 `main` 这种会变化的浮动版本作为实验唯一标识。
- `max_length=512`，截断策略固定并写入 manifest。
- 对标题与摘要使用稳定的拼接策略；建议优先保留标题，再截断摘要。
- `model.eval()`、`torch.inference_mode()`。
- 第一版不反向传播、不更新 FinBERT 参数。
- 批量推理，支持 CPU/GPU，支持断点续跑。
- 对每个 `source_id + input_text_hash + model_revision + tokenizer_revision + preprocessing_version` 建立缓存键。
- 模型下载后可离线运行；训练和面板构建不得依赖在线 API。
- 记录模型权重 SHA256、tokenizer revision、Transformers/PyTorch 版本、最大长度、池化方式和推理时间。

### 5.5 第一版文章级输出

每篇文章生成一行情绪数据：

| 字段 | 类型 | 说明 |
|---|---|---|
| `source_id` | string | 文章来源稳定 ID |
| `input_text_hash` | string | 标题+摘要标准化哈希 |
| `model_id` | string | `ProsusAI/finbert` |
| `model_revision` | string | 固定模型 commit |
| `tokenizer_revision` | string | 固定 tokenizer commit |
| `language` | string | 实际识别语言 |
| `token_count` | integer | 输入 token 数 |
| `p_positive` | float32 | 正面概率 |
| `p_negative` | float32 | 负面概率 |
| `p_neutral` | float32 | 中性概率 |
| `embedding_row` | integer | 文章向量文件行号 |
| `embedding_dim` | integer | 768 |
| `finbert_completed_at` | timestamp | 实际推理完成时间 |
| `finbert_status` | string | `success` / `unsupported_language` / `empty_text` / `error` |
| `schema_version` | string | 输出 Schema 版本 |

文章级 768 维向量单独保存，避免 Parquet 每行重复存储超长数组：

```text
article_embeddings_fp16.npy   # shape [M, 768]
article_finbert.parquet       # source_id -> embedding_row + sentiment + metadata
```

向量行必须通过 `embedding_row` 和 `source_id` 对应；文件重建时必须重新生成并校验索引。

### 5.6 768 维向量的固定提取方式

第一版建议使用最后一层 Transformer hidden state 的 `[CLS]` 向量：

\[
e_a = H^{last}_{a,\mathrm{CLS}}\in\mathbb{R}^{768}
\]

具体实现可从 `AutoModelForSequenceClassification` 的 `output_hidden_states=True` 输出中读取最后一层 hidden states。必须做单元测试确认：

- `embedding.shape == (batch_size, 768)`。
- `sentiment_probs.shape == (batch_size, 3)`。
- 概率有限且在 `[0,1]` 内。
- 每行概率和约等于 1。
- 输出顺序固定为 positive / negative / neutral。

不要在不同实验中混用 CLS pooling、mean pooling、pooler output。若未来比较 mean pooling，应作为新 `embedding_method` 和新特征版本保存。

### 5.7 情绪不是收益方向

FinBERT 的 positive / negative / neutral 是文本情绪分类结果：

- `positive` 不等于股票未来上涨。
- `negative` 不等于股票未来下跌。
- neutral 不等于没有风险。
- 新闻对不同公司、行业的影响可能相反。
- 公司业绩数字、宏观新闻、政策公告和市场评论的情绪含义也不同。

必须将情绪概率作为模型输入特征，由下游训练学习其与未来收益的关系；不得将情绪标签当作未来收益标签或手工买卖信号。

---

## 6. 文章向量压缩与训练集拟合约束

### 6.1 为什么需要压缩

若将 768 维文章向量直接为每只股票、每个交易日、每个通道保存成稠密 float32 面板，NASDAQ / NYSE 的存储量会明显增大。第一版建议：

1. 文章级保存原始 768 维 FinBERT 向量（float16）。
2. 在每个数据集的训练区间内拟合 `PCA(n_components=32)`。
3. 使用训练集拟合出的 PCA 转换训练、验证和测试文章向量。
4. 日度面板只保存 32 维压缩向量 + 情绪概率 + 少量统计特征。

PCA 只用于压缩，不使用收益率标签。

### 6.2 PCA 的防泄漏规则

- NASDAQ 的 PCA 只用 NASDAQ 训练区间内、按当前时间模式允许进入训练的文章拟合。
- NYSE 的 PCA 只用 NYSE 训练区间内允许进入训练的文章拟合。
- S&P 500 的 PCA 只用 S&P 500 训练区间内允许进入训练的文章拟合。
- 不得使用验证/测试文章拟合 PCA。
- 不得用全时期文章拟合后再声称没有未来信息泄漏。
- `StandardScaler`、缺失值填补器和其他统计变换同样只能在训练集拟合。
- 如果训练区间内文章数量不足以稳定拟合 PCA，停止 PCA 并报告；可改用固定随机投影作为无拟合备选，但必须记录 seed 和投影矩阵哈希。
- 保存 `pca_fit_start`、`pca_fit_end`、训练文章数、模型哈希和 `fit_mode`。

PCA 产物建议：

```text
data_llm/
  transforms/
    NASDAQ/
      finbert_pca_32.joblib
      transform_manifest.json
    NYSE/
      finbert_pca_32.joblib
      transform_manifest.json
    SP500/
      finbert_pca_32.joblib
      transform_manifest.json
```

---

## 7. 公司、行业、市场三通道构建

三通道要分别构建、分别保存 mask，不能将缺失新闻填成 neutral 新闻。

### 7.1 Company-level：公司级通道

只接收能够映射到当前数据集股票轴的公司新闻：

1. 文章实体必须是公司/发行人/证券相关实体。
2. `ticker`、`exchange`、`security_id` 必须来自第一部分的已验证实体或可靠证券主数据映射。
3. 通过数据集专属 `stock_uid -> array_index` 映射到当前股票数组。
4. 对历史新闻必须检查证券身份在事件日期/决策日期的有效期。
5. 对 ticker 更名、退市、并购、同名公司和多类证券，必须通过历史证券主数据解决。
6. 无法唯一映射的实体保留在文章表中，但不进入该数据集的公司级股票面板。
7. 同一文章涉及多个已验证公司时，可分别关联多个股票，但文章只保留一条主记录。

禁止：

- 仅根据公司名称让 LLM 猜 ticker。
- 仅因股票代码与公司名出现在同一段就自动绑定。
- 用当前 ticker 映射过去的 ticker。
- 把未上市公司、品牌、子公司自动当作上市股票。
- 把未映射公司随机匹配到相似公司。

### 7.2 Industry-level：行业级通道

行业新闻不需要文章明确写出 ticker，但必须有可核验的行业依据：

1. 使用第一部分 `article_industries` 和 `article_industry_impacts`。
2. 将原始行业名称映射到已确认的标准行业体系及版本。
3. `explicit` 与 `inferred` 分开保存。
4. 使用历史有效的 `security_industry_history`，将行业事件映射到在对应日期属于该行业的股票。
5. 行业分类必须按历史有效期查询，不能用今天的分类回填全部历史。
6. 一个股票属于多个行业/子行业时，按已验证映射规则处理；不能重复无控制地累加同一篇文章。
7. 如果没有可靠行业代码或历史行业映射，文章保留但不进行股票扩散。
8. 影响角色、方向和机制来自 Qwen 的描述性推断，不是收益率标签。

第一版建议对 `explicit` 和 `inferred` 分别记录并作为面板质量特征。不要默认将 inferred 当成 explicit。是否对 inferred 降权应作为验证集可调参数，不得在测试集调参。

### 7.3 Market-level：市场级通道

市场级新闻包括宏观数据、央行政策、财政政策、广泛市场制度、系统性风险等：

- `impact_scope` 必须包含 `market`，且有文章内容依据。
- 新闻市场归属应为 `US` 或 `GLOBAL` 才默认进入美国股票数据集。
- 市场级向量在同一数据集、同一交易日对所有股票共享。
- 面板可保存为 `[T, F]`，模型输入时再广播为 `[N, F]`。
- 不要把美国单公司新闻因为其网站或来源在美国就变成市场级新闻。
- 不要把中国市场新闻默认广播给 NASDAQ / NYSE / S&P 500。
- 对跨市场传导的规则留作后续版本，不在第一版随意加入。

### 7.4 三通道 mask

每个通道都有独立有效性 mask：

- `company_news_mask`
- `industry_news_mask`
- `market_news_mask`

含义：

- `1`：当前股票/市场在当前决策时点存在至少一篇符合时间、语言、映射和质量要求的新闻。
- `0`：没有符合条件的新闻或无法可靠映射。

没有新闻时，特征向量填 0，mask 为 0；不能将没有新闻解释为 FinBERT neutral。模型必须用 mask 强制该通道贡献为零。

---

## 8. 历史证券轴、行业轴和 S&P 500 成分的严格映射

### 8.1 `array_index` 是数据集局部编号

同一证券在 NASDAQ、NYSE、SP500 中可能有不同的数组行号。必须使用数据集专属映射：

```text
dataset
stock_uid
ticker
exchange
array_index
valid_from
valid_to
known_from
known_to
mapping_source
verification_status
```

强制规则：

- `array_index` 不能由 ticker 字母顺序推算。
- 不能假设 NASDAQ 和 NYSE 的股票数组顺序相同。
- 不能把 S&P 500 的数组行号复用为 NASDAQ / NYSE 行号。
- 每个数据集必须验证 `array_index` 恰好覆盖 `0 ... N-1`，无重复、无缺失。
- `stock_uid` 是稳定证券身份标识，不是数组下标。
- 若本地仓库缺少股票顺序映射，先从原始数据来源恢复并核验；无法核验时停止新闻面板生成，不得猜测。

### 8.2 S&P 500 历史成分偏差

论文报告 S&P 500 数据包含 474 只股票，但仅凭数据集名称和论文统计，不能证明其成分名单完整保留了每个历史日期的指数成分状态。

必须检查本地原始 S&P 500 数据的股票名单来源：

- 如果有逐日历史成分成员关系，使用 `valid_from/valid_to` 和 `known_from/known_to`。
- 如果只有固定 474 只股票名单，不得宣称它等于每个历史日期真实的 S&P 500 指数成分。
- 如果名单来自今天的成分股回填历史，必须标记潜在 survivorship bias。
- 不允许将退市、被剔除、并购或破产的历史公司静默删除。
- 新闻映射只能落到当前数据集确实包含且该时点有效的股票行。

### 8.3 历史行业映射

`security_industry_history` 应至少包含：

```text
stock_uid
industry_standard
industry_code
industry_name
valid_from
valid_to
known_from
known_to
mapping_source
verification_status
```

行业体系如 GICS、NAICS、SIC 必须明确名称、版本和映射来源。不能把不同体系的行业代码直接当成同一代码。若历史行业数据缺失，允许股票在对应日期没有行业新闻映射；不允许使用当前行业分类伪造历史分类。

---

## 9. 日度新闻特征面板

### 9.1 推荐目录

以下为建议结构，按本地仓库已有目录风格调整：

```text
D:\finance\model\StockMixer+llm\
│
├── src\
│   ├── model.py                         # 原始 StockMixer，尽量保持不变
│   ├── train.py                         # 原始基线入口，保留
│   ├── load_data.py                     # 原始行情加载逻辑，保留
│   ├── finbert_encoder.py               # 新增：FinBERT 离线推理
│   ├── news_input_loader.py             # 新增：读取第一部分 JSON/Parquet
│   ├── stock_axis_mapper.py             # 新增：数据集股票轴与历史映射
│   ├── news_panel_builder.py            # 新增：三通道 PIT 聚合
│   ├── news_dataset.py                  # 新增：新闻面板与行情窗口对齐
│   ├── model_with_news.py               # 新增：StockMixer + News Fusion
│   ├── train_with_news.py               # 新增：融合训练入口
│   └── evaluate_news.py                 # 新增：对照与消融评估
│
├── data_llm\
│   ├── input\                           # 第一部分正式摘要/Parquet 输入，不放原文
│   ├── article_features\
│   │   ├── article_finbert.parquet
│   │   ├── article_embeddings_fp16.npy
│   │   └── embedding_index.parquet
│   ├── mappings\
│   │   ├── stock_axis_NASDAQ.parquet
│   │   ├── stock_axis_NYSE.parquet
│   │   ├── stock_axis_SP500.parquet
│   │   ├── security_master.parquet
│   │   └── security_industry_history.parquet
│   ├── transforms\
│   │   ├── NASDAQ\
│   │   ├── NYSE\
│   │   └── SP500\
│   ├── panels\
│   │   ├── NASDAQ\
│   │   ├── NYSE\
│   │   └── SP500\
│   ├── manifests\
│   └── logs\
│
├── checkpoints\
│   ├── baseline\
│   └── news_fusion\
│
└── configs\
    ├── datasets.yaml
    ├── finbert.yaml
    └── news_fusion.yaml
```

如果原始行情 `dataset/` 已经存在，保留其原目录，不要将其复制或移动到 `data_llm`。`data_llm` 只存新闻输入、文本特征、映射快照、压缩变换、新闻面板和 manifest。

### 9.2 日度面板 shape

默认特征版本 `news_feature_schema=v1`：

- 公司级面板：`[N, T, 40]`
- 行业级面板：`[N, T, 40]`
- 市场级面板：`[T, 40]`
- 公司 mask：`[N, T]`
- 行业 mask：`[N, T]`
- 市场 mask：`[T]`

其中：

| 特征段 | 维度 | 含义 |
|---|---:|---|
| FinBERT sentiment | 3 | `p_positive`, `p_negative`, `p_neutral` |
| FinBERT PCA | 32 | 训练集拟合 PCA 后的文章向量加权聚合 |
| `log1p_article_count` | 1 | 当前聚合窗口内文章数量 |
| `mean_age_sessions` | 1 | 文章距当前决策日的平均交易日数 |
| `inferred_share` | 1 | 行业通道中 inferred 关系占比 |
| `mean_importance` | 1 | 第一部分有可靠 importance 字段时的均值 |
| `importance_available_ratio` | 1 | 有 importance 的文章占比 |

总维度为 `3 + 32 + 5 = 40`。

如果第一部分没有 importance 字段，不能伪造 importance；对应值按 Schema 约定填充，并由 `importance_available_ratio=0` 表示该元数据缺失。不同版本的特征维度变化必须升级 `news_feature_schema`，不得复用旧 manifest。

### 9.3 面板文件名

每个数据集输出：

```text
company_news_daily.npy       # [N, T, 40]
industry_news_daily.npy      # [N, T, 40]
market_news_daily.npy        # [T, 40]

company_news_mask.npy        # [N, T]
industry_news_mask.npy       # [N, T]
market_news_mask.npy         # [T]

trading_dates.npy            # [T]，或保存为可读日期表
stock_axis.parquet           # array_index -> stock_uid/ticker
manifest.json
```

建议用 float32 存储面板，或先用 float16 存储、加载时转 float32；必须通过数值稳定性测试。大型 `.npy` 文件可采用 memory map 读取。

### 9.4 文章聚合窗口

第一版采用因果滚动窗口：

- 仅聚合决策时点前已可用文章。
- 默认最多回看最近 5 个交易日。
- 文章年龄按交易日序号计算，不按自然日简单相减。
- 默认时间权重：

\[
w(a,t)=\exp(-age\_sessions(a,t)/\tau)
\]

其中 `tau` 初始可设为 3 个交易日，最终只能依据验证集选择。任何 `age_sessions < 0` 的文章都必须被拒绝。

对每个通道：

- sentiment 概率做加权平均。
- 32 维 PCA embedding 做加权平均。
- 文章数、平均年龄、inferred 比例等统计量单独保存。
- 不同文章的权重必须有明确来源；缺少 source quality / importance 时使用中性默认权重，不要让模型自评 confidence 直接变成真实概率。
- 文章重复按 `source_id` 和内容哈希去重；跨来源事件聚类只能使用决策时点已知的聚类结果。

### 9.5 缺失值与 mask

- 没有文章：面板向量全 0、mask=0。
- 有文章但语言不支持 FinBERT：FinBERT 特征 mask=0，并记录原因；不得伪造 neutral。
- 有文章但无法映射股票：保留文章级特征，不进入公司/行业股票面板。
- 公司通道无新闻，不影响行业或市场通道。
- 行业通道无新闻，不影响公司或市场通道。
- 市场通道无新闻，不影响公司或行业通道。
- 新闻 mask 不能与原始 `mask_data` 混为一谈；行情样本有效性和新闻可用性是两个独立概念。

---

## 10. 三通道 Gated Residual Fusion：第一版模型结构

### 10.1 设计目标

第一版采用轻量 Late Fusion：

- 原始 StockMixer 负责行情时序和股票间关系建模。
- 三个新闻分支分别编码公司、行业和市场新闻。
- 每个分支使用独立 Adapter 和 Gate。
- 新闻分支只输出对基础预测的残差修正。
- mask 为 0 时，该通道贡献必须严格为 0。
- 不改变原始 `get_loss()` 的收益率 MSE + ranking loss 定义。

### 10.2 模型输入

一个交易日的模型输入建议为：

```text
stock_x:             [N, lookback=16, F_stock]
base_price:          [N, 1]

company_news:        [N, 40]
industry_news:       [N, 40]
market_news:         [N, 40]   # 将 [40] 广播给 N 只股票，或在 loader 中直接展开

company_mask:        [N, 1]
industry_mask:       [N, 1]
market_mask:         [N, 1]
```

其中 `N` 必须等于当前数据集 `eod_data.shape[0]`，而不是写死的股票数。

### 10.3 融合公式

原始 StockMixer 输出基础预测价格：

\[
\widehat P^{SM}_{i,t+1}
\]

转换成基础预测收益率：

\[
\widehat r^{SM}_{i,t+1}
=
\frac{\widehat P^{SM}_{i,t+1}-P_{i,t}}{P_{i,t}}
\]

对每个新闻通道 \(k\in\{c,i,m\}\)：

\[
h^k_i = Adapter_k(News^k_i)
\]

\[
g^k_i =
\sigma(Gate_k([h^k_i,\widehat r^{SM}_{i,t+1}]))
\cdot mask^k_i
\]

\[
\Delta r^k_i = g^k_i \cdot Head_k(h^k_i)
\]

最终：

\[
\widehat r^{Fusion}_{i,t+1}
=
\widehat r^{SM}_{i,t+1}
+
\Delta r^c_i+
\Delta r^i_i+
\Delta r^m_i
\]

再还原为预测价格：

\[
\widehat P^{Fusion}_{i,t+1}
=
P_{i,t}(1+\widehat r^{Fusion}_{i,t+1})
\]

将融合后的预测价格交给原始 `get_loss()`，保持标签和损失口径一致。

### 10.4 初始化和缺失新闻处理

- 三个 `Adapter` 独立。
- 三个 `Gate` 独立。
- 三个 `Head` 独立。
- `Head` 最后一层权重和 bias 建议初始化为 0，使初始融合输出与基础 StockMixer 完全一致。
- Gate bias 可初始化为负值，让模型从较小新闻贡献开始学习。
- 每个通道输出都乘以对应 mask。
- 当 mask=0 时，必须严格保证 `delta=0`，而不是只依赖训练学习忽略空向量。
- 新闻特征缺失不能影响基础 StockMixer 的正常预测。
- 不得因没有新闻而丢弃有效行情样本。

### 10.5 推荐的模型包装方式

建议新增 `StockMixerWithNews`，内部组合现有 `StockMixer`：

```python
class StockMixerWithNews(nn.Module):
    def __init__(self, stock_mixer, news_dim=40, hidden_dim=32):
        super().__init__()
        self.stock_mixer = stock_mixer

        # 为 company / industry / market 各建独立 Adapter、Gate、Delta Head
        # 具体实现按本地 StockMixer 输出 shape 编写

    def forward(
        self,
        stock_x,
        base_price,
        company_news,
        industry_news,
        market_news,
        company_mask,
        industry_mask,
        market_mask,
    ):
        # 1. 原始 StockMixer 预测价格
        # 2. 转换成基础预测收益率
        # 3. 计算三个通道的 gated residual
        # 4. mask 强制屏蔽无新闻通道
        # 5. 转回预测价格，保持原 get_loss 接口
        ...
```

这是接口示意，不是要求原样复制。Coding Agent 必须根据本地 `StockMixer.forward()` 的实际输出 shape 适配，并通过单元测试验证。

### 10.6 不要在第一版重构 StockMixer 主干

- 保留原始 `model.py` 中的 StockMixer 实现。
- 不把 FinBERT 模型放进 StockMixer。
- 不把文章级 768 维文本向量直接拼接到行情 `F_stock` 维度。
- 不改原始 stock mixing 的股票数量和参数语义。
- 不将新闻通道直接塞入原始 `NoGraphMixer` 的股票轴矩阵。
- 不改变原始收益率标签、pairwise rank loss 和行情 mask。
- 融合模型可以单独保存 checkpoint，基线 checkpoint 不覆盖。

---

## 11. 训练策略与消融实验

### 11.1 推荐训练顺序

**Step 1：先训练或加载纯 StockMixer 基线**

- 完全使用原始行情输入和原始损失函数。
- 每个数据集单独训练、单独保存 checkpoint。
- 仅使用训练集更新参数。
- 仅使用验证集选择 epoch / checkpoint。
- 测试集在 checkpoint 锁定后评估一次。
- 如果仓库已有可信 baseline checkpoint，先验证配置和数据版本一致，再决定是否复用。

**Step 2：初始化新闻融合模型**

- 将 baseline checkpoint 加载到 `StockMixerWithNews.stock_mixer`。
- 初始新闻 residual 为零，确认融合模型初始预测与 baseline 完全一致。
- 默认先冻结 StockMixer 主干，只训练三个 Adapter、Gate 和 Delta Head。
- 使用训练集拟合参数，验证集早停和选模。

**Step 3：可选联合微调**

如果 Step 2 有稳定增益，再进行联合微调：

- StockMixer 主干使用较小学习率。
- 新闻 Adapter / Gate 使用较大学习率。
- 仍然只使用验证集选模。
- 如果联合微调不稳定或验证效果下降，保留冻结主干版本。

### 11.2 必须做的消融

| 实验 | 模型 | 目的 |
|---|---|---|
| E0 | 原始 StockMixer | 纯行情基线 |
| E1 | StockMixer + company | 公司级新闻增量 |
| E2 | StockMixer + industry | 行业级新闻增量 |
| E3 | StockMixer + market | 市场级新闻增量 |
| E4 | StockMixer + 三通道 | 完整第一版 |
| E5 | 三通道，仅 FinBERT 情绪 3 维 | 验证情绪概率贡献 |
| E6 | 三通道，仅 PCA embedding 32 维 | 验证语义向量贡献 |
| E7 | 三通道，情绪 + embedding | 验证组合效果 |
| E8 | 随机错位新闻日期 | 泄漏/伪相关负对照 |
| E9 | 随机股票映射 | 验证行业传播是否只是噪声 |

E8 / E9 仅用于负对照研究，不作为候选正式模型。

### 11.3 评价指标

尽量沿用原始 StockMixer 评价，并补充时间序列和交易评价：

- MSE / MAE。
- IC（每日截面 Pearson 相关系数）。
- RankIC / RIC（每日截面 Spearman 相关系数）。
- ICIR / RankICIR（明确是否年化和年化方式）。
- Precision@10。
- Sharpe Ratio（明确收益频率、无风险利率、年化因子）。
- Maximum Drawdown。
- Turnover。
- 交易成本后的收益表现（只有定义执行时点、成本和滑点后才报告）。

所有实验必须使用相同的数据集、股票轴、日期切分、标签、有效股票 mask、选模规则和测试区间。至少使用 3 个随机种子，报告均值和标准差。

### 11.4 不得根据测试集选参数

以下内容只允许根据训练/验证集选择：

- PCA 维度（第一版固定 32，若要比较也只用验证集）。
- 新闻时间衰减参数 `tau`。
- 新闻回看交易日数。
- Gate/Adapter 隐藏层维度。
- dropout。
- 学习率和 early stopping。
- explicit / inferred 行业新闻权重。
- 是否联合微调 StockMixer。
- 各通道开关。

测试集只在最终方案锁定后运行一次。任何看过测试集结果后进行的方案修改，都需要新的未触碰测试区间或明确标记为探索性分析。

---

## 12. 数据集与新闻面板对齐接口

### 12.1 每个数据集必须独立生成

必须分别生成：

```text
NASDAQ
NYSE
SP500
```

每个数据集拥有独立：

- `stock_axis`
- `trading_dates`
- `dataset_split`
- `historical_security_mapping`
- `historical_industry_mapping`
- `PCA`
- `company_news_daily`
- `industry_news_daily`
- `market_news_daily`
- masks
- manifest

即使 NASDAQ 和 NYSE 日期轴相同，也不能混用股票数组行号。

### 12.2 面板与行情 shape 断言

训练前至少执行：

```python
assert eod_data.ndim == 3
assert eod_data.shape[0] == stock_axis.shape[0]
assert eod_data.shape[1] == len(trading_dates)

assert mask_data.shape == eod_data.shape[:2]
assert gt_data.shape == mask_data.shape
assert price_data.shape == mask_data.shape

assert company_news_daily.shape[:2] == mask_data.shape
assert industry_news_daily.shape[:2] == mask_data.shape
assert market_news_daily.shape[0] == mask_data.shape[1]

assert company_news_mask.shape == mask_data.shape
assert industry_news_mask.shape == mask_data.shape
assert market_news_mask.shape == (mask_data.shape[1],)
```

同时检查：

- `stock_axis.array_index` 连续、唯一。
- 面板股票行顺序 hash 与行情数组股票顺序 hash 完全一致。
- 面板日期顺序 hash 与行情日期顺序 hash 完全一致。
- 所有浮点值有限，无 NaN / Inf。
- mask 只能为 0 / 1。
- 无新闻位置的特征为 0。
- 每个面板都记录相同的 `dataset_id`、`stock_axis_hash`、`trading_dates_hash`。

任何断言失败都应停止训练，而不是自动 reshape、排序或截断。

### 12.3 训练样本读取

对每个样本：

```text
offset = s
decision_idx = s + lookback_length - 1
target_idx = decision_idx + steps

stock_x = eod_data[:, s : s + lookback_length, :]
base_price = price_data[:, decision_idx]
target_return = gt_data[:, target_idx]

company_news = company_news_daily[:, decision_idx, :]
industry_news = industry_news_daily[:, decision_idx, :]
market_news = market_news_daily[decision_idx, :]

company_mask = company_news_mask[:, decision_idx]
industry_mask = industry_news_mask[:, decision_idx]
market_mask = market_news_mask[decision_idx]
```

`decision_idx` 是新闻特征截止日期，`target_idx` 是未来标签日期。**不允许用 `target_idx` 或更晚日期的新闻面板作为输入。**

如果原始 `get_batch()` 的实现与上述索引不完全相同，先核对它的标签定义和价格索引，再进行等价适配；不得只按变量名称猜测。

---

## 13. 配置文件建议

配置名称可按本地项目规范调整：

```yaml
paths:
  stockmixer_data_root: "../dataset"
  news_data_root: "D:/finance/model/StockMixer+llm/data_llm"
  news_input_dir: "D:/finance/model/StockMixer+llm/data_llm/input"

finbert:
  model_id: "ProsusAI/finbert"
  model_revision: "PINNED_COMMIT_REQUIRED"
  tokenizer_revision: "PINNED_COMMIT_REQUIRED"
  max_length: 512
  pooling: "cls_last_hidden"
  hidden_size: 768
  freeze: true
  batch_size: 16
  use_fp16_embeddings: true
  supported_languages:
    - "en"

news_features:
  schema_version: "v1"
  pca_components: 32
  max_lookback_sessions: 5
  decay_tau_sessions: 3.0
  company_channel: true
  industry_channel: true
  market_channel: true
  missing_news_is_zero_with_mask: true

time_alignment:
  mode: "retrospective"  # observed_pit / simulated_pit / retrospective
  timezone: "America/New_York"
  date_only_policy: "next_trading_session"
  require_feature_available_at: true
  require_model_available_at: true
  require_mapping_known_at_for_observed_pit: true

fusion:
  news_dim: 40
  hidden_dim: 32
  initialize_delta_head_zero: true
  freeze_backbone_first: true
  allow_joint_finetune: false

datasets:
  NASDAQ:
    train_end_idx: 756
    valid_end_idx: 1008
    expected_stock_count: 1026
    market_num: 20

  NYSE:
    train_end_idx: 756
    valid_end_idx: 1008
    expected_stock_count: 1737
    market_num: 25

  SP500:
    train_end_idx: 1006
    valid_end_idx: 1259
    expected_stock_count: 474
    market_num: 8
```

以上是配置模板，不是可直接无检查运行的配置。模型 revision 必须替换成实际固定 commit；数据集边界必须先与本地真实日期轴核对。`mode=retrospective` 仅是示例默认值，不能被写成严格 PIT。

---

## 14. 推荐依赖与运行要求

根据本地环境选择兼容版本，不要为了安装新依赖而升级破坏原始 StockMixer 的 PyTorch 环境。

可能需要：

- `transformers`
- `torch`
- `numpy`
- `pandas`
- `pyarrow`
- `scikit-learn`
- `safetensors`
- `joblib`
- 交易日历库（如项目已有则复用）

要求：

1. 记录 Python、PyTorch、Transformers、NumPy、Pandas、CUDA 版本。
2. 模型权重下载后固定本地缓存目录。
3. 支持完全离线推理。
4. 不在每个训练 epoch 重复运行 FinBERT。
5. 使用 batch inference 和断点缓存。
6. 对 GPU OOM 支持降低 batch size 后重试。
7. 模型输出写入临时文件，校验成功后原子替换正式文件。
8. 推理失败的文章记录错误代码，可安全重试。
9. 不把文章全文写进日志、异常信息或 debug 文件。
10. 对模型输出和面板进行版本化，防止不同模型版本的向量混用。

---

## 15. 模块职责建议

按现有仓库结构映射，不要求机械创建同名文件。

| 模块 | 职责 |
|---|---|
| `finbert_encoder.py` | 加载固定 FinBERT、英文校验、情绪分类、768 维 CLS 向量、缓存 |
| `news_input_loader.py` | 读取第一部分 JSON/Parquet，Schema 兼容和时间字段校验 |
| `stock_axis_mapper.py` | 读取 dataset-specific 股票顺序、证券主数据和历史映射 |
| `news_panel_builder.py` | PIT 过滤、三通道映射、时间衰减、日度面板聚合 |
| `news_dataset.py` | 将 `[N,T,F]` 行情与三通道新闻面板按决策日对齐 |
| `model_with_news.py` | 包装原始 StockMixer，增加三个 Adapter/Gate/Delta Head |
| `train_with_news.py` | baseline checkpoint 加载、冻结训练、验证集选模、最终测试 |
| `evaluate_news.py` | E0-E9 消融、指标、覆盖率和泄漏检查 |
| `manifest.py` | 记录模型、数据、日期轴、股票轴、Schema、时间模式和哈希 |

禁止：

- 将全部新逻辑塞进原 `train.py`。
- 将 FinBERT 加载放在 `model.forward()` 中。
- 在训练循环内读取 Parquet 并逐篇运行模型。
- 在 `model.py` 中硬编码 `1026`、`1737`、`474` 以外的新闻维度假设。
- 用 `ticker` 直接作为数组下标。
- 将不同数据集的新闻面板混在一个无 dataset 标识的文件中。

---

## 16. 必须通过的测试

### 16.1 FinBERT 单元测试

1. 模型固定 revision 可加载。
2. 输出标签顺序固定为 positive / negative / neutral。
3. 情绪概率 shape 为 `[B,3]`，和约等于 1。
4. 768 维 embedding shape 为 `[B,768]`。
5. 同一文本、同一模型版本重复运行结果一致（允许合理浮点误差）。
6. 中文文章被标记 unsupported，不伪装为英文预测。
7. 空标题、空摘要、超长摘要、特殊字符能安全处理。
8. 不保存输入全文。
9. 断点缓存命中时不重复推理。
10. 模型 revision 或输入 hash 变化时正确重新推理。

### 16.2 数据集和日期轴测试

1. NASDAQ 股票数与行情数组一致，论文版本预期 1026。
2. NYSE 股票数与行情数组一致，论文版本预期 1737。
3. SP500 股票数与行情数组一致，论文版本预期 474。
4. NASDAQ/NYSE 日期范围与本地源文件核对。
5. SP500 切片起点必须由日期元数据确认，不能只信任 915。
6. 每个数据集股票行号唯一、连续、可回溯到 `stock_uid`。
7. 新闻面板日期 hash 与行情日期 hash 一致。
8. 面板的 N、T、F 维度完全匹配。

### 16.3 PIT 测试

构造合成文章：

- 决策时间前发布、处理完成 → 可进入。
- 决策时间前发布、决策时间后才抓取 → 不可进入。
- 决策时间前发布、FinBERT 决策时间后才完成 → 不可进入。
- 收盘后发布 → 下一交易日或后续可用时点。
- 只有日期的发布时间 → 下一交易日。
- 时区跨夏令时 → 正确转换。
- 周末或节假日文章 → 映射到下一个可用交易日。
- `age_sessions < 0` → 必须拒绝。
- 没有真实 first_seen/fetched 时间 → 不得标记 observed_pit。
- `availability_basis=simulated` → 结果 manifest 必须标记 simulated_pit。
- FinBERT 模型发布日晚于决策日 → observed_pit 拒绝。
- 历史行业映射 known_from 晚于决策日 → observed_pit 拒绝。

### 16.4 股票与行业映射测试

1. 同一 ticker 在不同数据集可以有不同 `array_index`。
2. ticker 更名或复用时按有效期匹配。
3. 同名公司不得自动合并。
4. 无 ticker 的公司实体不进入公司股票面板。
5. 未验证证券映射不进入公司通道。
6. explicit 和 inferred 行业关系分开记录。
7. 当前行业分类不得自动回填历史。
8. 没有历史行业映射的股票不接收该行业新闻。
9. S&P 500 成分历史缺失时标记潜在 survivorship bias。
10. 文章只保留一个主记录，不因映射多个股票而复制文章记录。

### 16.5 Fusion 模型测试

1. 所有新闻 mask=0 时，融合模型输出必须与基础 StockMixer 一致。
2. 只有 company mask=1 时，industry/market delta 必须为 0。
3. 只有 industry mask=1 时，company/market delta 必须为 0。
4. 只有 market mask=1 时，company/industry delta 必须为 0。
5. 新闻特征 shape 错误时抛出清晰异常。
6. `base_price` 为零或非法值时按原数据质量规则处理，不产生 Inf。
7. 融合输出 shape 与原模型一致。
8. 原始 `get_loss()` 可以直接计算融合输出损失。
9. Delta Head 零初始化时输出与 baseline 相同。
10. 没有新闻时不会因为新闻 mask 导致有效行情样本被丢弃。

### 16.6 训练与泄漏测试

1. scaler / PCA 只在训练集拟合。
2. 验证/测试数据不参与拟合 PCA。
3. 测试集不参与 early stopping 或超参数选择。
4. 训练样本 target index 不跨越训练边界。
5. 验证/测试样本只使用决策日及以前的新闻。
6. 同一训练配置和 seed 可复现。
7. 不同 dataset 的股票轴和日期轴不能混用。
8. 训练期间不调用 FinBERT。
9. 测试集最终只评估一次。
10. E0 的无新闻输出与原始 StockMixer 基线在容差范围内一致。

---

## 17. 分阶段实施顺序

### Phase 0：只读盘点

- 检查用户本地仓库。
- 核对 `model.py`、`train.py`、`load_data.py`、`evaluator.py` 的实际路径。
- 检查 NASDAQ / NYSE `.pkl` 和 SP500 `.npy` 的真实 shape。
- 找到股票 ticker / stock_uid / array_index 对照文件。
- 恢复真实交易日期轴。
- 核对三数据集切分边界。
- 检查当前 `get_batch()` 的标签索引和预测价格口径。
- 输出风险报告，不改代码。

### Phase 1：FinBERT 离线编码

- 固定 FinBERT 模型 revision。
- 检查许可证。
- 读取第一部分摘要和标题。
- 仅处理英文文章。
- 输出情绪概率、768 维向量和文章级 Parquet。
- 加入 hash 缓存、断点续跑和失败记录。
- 不生成日度股票面板。

### Phase 2：股票/行业映射与时间协议

- 建立三个 dataset-specific stock axis。
- 校验历史证券映射和行业映射。
- 建立 observed_pit / simulated_pit / retrospective 三种时间模式。
- 建立交易日历和日期-only 新闻顺延逻辑。
- 加入模型发布日期和映射 known_from 检查。
- 先用合成数据验证 PIT。

### Phase 3：三通道日度面板

- 构建 company / industry / market 三通道。
- 每个通道独立 mask。
- 使用训练集拟合 PCA。
- 按决策时点过滤新闻并滚动聚合。
- 分别生成 NASDAQ / NYSE / SP500 面板。
- 保存 manifest 和数据 hash。
- 验证 shape 与股票/日期轴完全一致。

### Phase 4：融合模型

- 新增 `StockMixerWithNews`。
- 载入原始 baseline checkpoint。
- 实现三个 Adapter/Gate/Delta Head。
- 验证零新闻时输出等于原始 StockMixer。
- 先冻结 StockMixer 主干训练新闻分支。
- 可选联合微调，只有验证集有稳定提升时保留。

### Phase 5：消融、测试和报告

- 跑 E0-E9。
- 使用相同数据切分和标签。
- 报告新闻覆盖率、语言覆盖率、映射成功率、PIT 可用率。
- 报告三个数据集各自的指标。
- 报告回溯性、模拟 PIT、真实 observed PIT 的区别。
- 测试集最终评估一次。
- 不得只报告最好的一次 seed。

---

## 18. 验收标准

### A. 原始 StockMixer 保留

- [ ] 原始模型和训练入口仍可运行。
- [ ] 原始基线 checkpoint 不被覆盖。
- [ ] 原始行情输入、标签、mask 和损失函数未被无意改变。
- [ ] E0 能复现本地原始 StockMixer 基线。

### B. FinBERT

- [ ] 使用固定 revision 的 `ProsusAI/finbert`。
- [ ] 许可证和用途经过检查。
- [ ] 英文输入策略明确，中文不伪装为有效 FinBERT 特征。
- [ ] 输出三类概率和 768 维文章向量。
- [ ] 推理离线运行，训练时不重复调用 FinBERT。
- [ ] 模型版本、tokenizer、文本 hash、推理时间有记录。

### C. 数据集

- [ ] NASDAQ / NYSE / SP500 分别有独立配置。
- [ ] 日期范围、股票数、日期轴和股票顺序均经过本地数据核验。
- [ ] SP500 不盲目照搬 756 / 1008。
- [ ] 股票 `array_index` 与行情数组行序完全一致。
- [ ] S&P 500 历史成分和 survivorship bias 有明确说明。

### D. 时间防泄漏

- [ ] `published_at` 与真实特征可用时间分离。
- [ ] 日期-only 文章保守顺延。
- [ ] 收盘后新闻不能进入当日收盘预测特征。
- [ ] FinBERT 模型发布时间纳入检查。
- [ ] 历史映射有 valid time / known time。
- [ ] PCA 和 scaler 仅训练集拟合。
- [ ] 按标签日期切分，不随机拆分。
- [ ] 测试集不参与选模，最终只测试一次。
- [ ] 回溯实验不冒充严格 observed PIT。

### E. 三通道与融合

- [ ] 公司、行业、市场三个通道分别生成。
- [ ] 三个通道各自有 mask。
- [ ] 没有新闻时通道贡献严格为零。
- [ ] 行业新闻只传播到历史有效且已映射的股票。
- [ ] 市场新闻共享给美国市场数据集时有市场归属过滤。
- [ ] 融合模型保留基础 StockMixer 输出作为 residual baseline。
- [ ] 原始损失函数可正常用于融合预测。
- [ ] E0-E9 消融可复现。

### F. 输出与报告

- [ ] `data_llm` 下生成文章级 FinBERT 特征。
- [ ] 生成三个数据集各自的三通道面板和 mask。
- [ ] 每个面板均有 manifest。
- [ ] 提供完整测试命令和通过/失败数量。
- [ ] 提供覆盖率、语言支持率、实体映射率、行业映射率和 PIT 可用率。
- [ ] 提供 NASDAQ / NYSE / SP500 分别的验证/测试结果。
- [ ] 明确列出无法达到严格 PIT 的数据区间和原因。

---

## 19. 完成后向项目负责人汇报

请按以下格式汇报：

1. **本地仓库盘点**：实际目录、入口文件、模型类、训练流程和数据加载方式。
2. **行情数据核验**：三个数据集的 shape、日期范围、股票数、股票顺序来源。
3. **切分核验**：每个数据集实际 train / val / test 日期边界和样本数。
4. **FinBERT 版本**：模型 revision、tokenizer revision、许可证、推理参数和模型 SHA256。
5. **文章特征统计**：总文章数、英文文章数、成功推理数、失败数、语言不支持数。
6. **股票映射统计**：公司新闻匹配率、行业新闻映射率、市场新闻覆盖率。
7. **PIT 统计**：observed_pit / simulated_pit / retrospective 各自可用文章数和覆盖率。
8. **面板 shape**：三个数据集每个通道的 shape、mask shape、hash。
9. **模型改造文件清单**：新增文件、修改文件、未修改的原始文件。
10. **训练结果**：E0-E9、三个随机种子、验证与最终测试指标。
11. **泄漏检查**：日期、模型版本、PCA、股票映射、行业映射、S&P 500 成分、测试集选模检查结果。
12. **未完成事项**：历史时间戳、历史指数成分、历史行业分类、FinBERT 非英文覆盖等。
13. **明确声明**：没有把历史回溯实验伪装成严格 PIT；没有修改或保存原始文章全文；原始 StockMixer 基线仍可独立运行。

---

## 20. 本阶段最终边界

第二部分完成后，系统应具备：

- 以第一部分正式摘要和结构化 Parquet 为输入的 FinBERT 离线编码流程。
- 每篇英文文章的三类情绪概率和 768 维语义向量。
- 训练集拟合的 32 维 PCA 压缩。
- 与 NASDAQ、NYSE、S&P 500 各自股票轴和日期轴严格对齐的公司/行业/市场新闻面板。
- 独立的新闻 mask 和 PIT 时间过滤。
- 历史证券、行业、指数成分映射的时间有效性检查。
- 不改写基础 StockMixer 的轻量 Gated Residual Fusion。
- 可复现的 baseline、融合训练、验证、测试和消融实验。

**最重要的结论：** 新闻特征必须在预测决策时点已经可用；模型版本、摘要版本、映射版本也必须在该时点可用。NASDAQ / NYSE 的历史区间早于 FinBERT 论文发布时间，因此第一版完整历史实验应明确作为回溯性或模拟延迟研究；不能宣称为真实历史实时 PIT。S&P 500 也必须逐段检查模型发布时间、新闻可用时间和历史成分映射。

---

## 参考资料

1. StockMixer 官方代码：<https://github.com/SJTU-DMTai/StockMixer>
2. StockMixer AAAI 2024 论文：<https://ojs.aaai.org/index.php/AAAI/article/view/28681>
3. StockMixer 论文 PDF（含 NASDAQ / NYSE / S&P 500 样本统计）：<https://ojs.aaai.org/index.php/AAAI/article/download/28681/29322>
4. ProsusAI FinBERT 模型卡（模型结构、语言、标签、许可证）：<https://huggingface.co/ProsusAI/finbert>
5. FinBERT 论文：<https://arxiv.org/abs/1908.10063>
6. NASDAQ / NYSE 原始数据论文引用来源：<https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>
7. S&P 500 数据来源相关项目：<https://github.com/thanhtrunghuynh93/estimate>
