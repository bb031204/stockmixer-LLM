# StockMixer 股票身份映射与自然语言数据对齐说明

## 1. 结论：股票身份是数组行位置，股票代码不直接输入模型

NASDAQ、NYSE 的 `eod_data.pkl` 和 SP500 的 `SP500.npy` 都是纯数值 `numpy.ndarray`，形状为 `[股票数, 日期数, 特征数]`，数组内没有公司名、股票代码或显式证券 ID 字段。模型以第 0 维的 **0 起始行索引 `array_index`** 隐式区分股票；它不是交易所证券编号，也不是跨数据集统一编号。

外部源文件提供 ticker（如 `AAPL`、`POOL`、`A`），可与数组索引逐行对应。文末列出了全部 **3237 条数据集内股票映射**，没有省略号或抽样。

| 数据集 | 数组形状 | 索引范围 | 股票顺序来源 |
|---|---|---|---|
| NASDAQ | `(1026,1245,5)`，float32 | 0~1025 | 官方合格 ticker 列表，已逐行核验全部 CSV |
| SP500 | `(474,2526,5)`，float64 | 0~473 | 原始字典的键顺序，已核验全部数值 |
| NYSE | `(1737,1245,5)`，float32 | 0~1736 | 官方合格 ticker 列表及已安装的 tickers.csv，已逐行核验全部 CSV |

**三个数据集的 `array_index=0` 分别是 `NASDAQ:AABA`、`SP500:POOL`、`NYSE:A`，并不是同一只股票。** 本文建议用 `stock_uid = dataset + ':' + ticker` 作为文本数据连接键。`stock_uid` 是本次新导出的元数据字段，不是原模型已有字段；它仅在本次数据集版本中定位股票，不代表跨历史永久证券 ID。

## 2. 顺序如何确认

- **NASDAQ**：按源 `NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv` 的顺序，将每只股票的 CSV 去掉最后一天、去掉索引列，再按原加载器将 `-1234` 替换为 `1.1`；全部 1026 行、全部日期及五列特征与现有 pickle 完全相同。
- **NYSE**：按源 `NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv` 顺序逐行对照，全部 1737 行与现有 pickle 完全相同；与 `dataset/NYSE/tickers.csv` 顺序也一致。
- **SP500**：`baseline_data_sp500.npy` 解包后是 `dict[ticker, DataFrame]`；对其键顺序对应的 `[open,high,low,volume,close]` 按每股、每列的全样本最大值归一化，结果与 `SP500.npy` 的全部元素完全一致。因此该键顺序就是当前数组的股票轴顺序。

**不要按 `sp500_ticker.csv` 的行号、公司名或 ticker 字母重新排序。** 该表有 505 行，而实际模型有 474 只股票；该表第一行是 `MMM`，实际数组第一行是 `POOL`。它只适合按 `Symbol` 连接名称/行业等元数据，不能直接提供数组行号。

训练代码直接读取整个股票轴，股票混合层的权重依赖这个固定顺序。若要重排股票轴，价格、标签、mask、文本向量及已有股票混合权重都必须一致处理，不能把新顺序直接套到旧检查点上。

## 3. 名称与 Wikidata ID 的覆盖范围

公司名仅取本地 `dataset/SP500/sp500_ticker.csv` 中的 `Name`，按 ticker **精确匹配**；这些名称是该文件的快照，尚未逐一核验样本期内更名、代码复用或公司重组，不作为历史法律实体身份的保证。没有本地名称时标为“未提供”，不会猜测补齐。股票代码在三套映射中均完整覆盖。

| 数据集 | 全部股票 | 本地名称可用 | 公司名未提供 | 源文件已给 Wikidata QID |
| NASDAQ | 1026 | 109 | 917 | 512 |
| SP500 | 474 | 474 | 0 | 0 |
| NYSE | 1737 | 308 | 1429 | 597 |

NASDAQ/NYSE 的 `*_wiki.csv` 另有 Wikidata 标识（如 `Q312`），源文件的 `unknown` 在本文用“—”表示。QID 是外部知识库实体辅助信息，不是数组行号，当前主干模型也没有读取它。不同证券/股类可能对应同一实体，不能按 QID 自动合并股票。SP500 没有专属 QID 表，本次不从其他年代的数据集中迁移 QID。

历史文本检索时应保留原 ticker，并另建 `historical_aliases`、`valid_from`、`valid_to` 等别名区间；不要用今天的公司名或 ticker 直接覆盖历史数组身份。尚未核验的名称只能作为检索候选。

## 4. 自然语言记录建议格式

以下是字段示例，不是真实新闻，也不表示已经实现文本模型：

```json
{
  "dataset": "SP500",
  "array_index": 0,
  "stock_uid": "SP500:POOL",
  "ticker": "POOL",
  "company_name": "Pool Corporation",
  "input_end_date": "2016-01-26",
  "target_date": "2016-01-27",
  "published_at": null,
  "available_at": null,
  "prediction_cutoff": null,
  "text": "此处填写截至预测时点已可获得的、与该股票或市场相关的文本。",
  "source": "填写真实来源或生成方法",
  "is_synthetic": true
}
```

`published_at`、`available_at` 和 `prediction_cutoff` 在真实训练样本中应填为带时区的时间，并要求 `available_at < prediction_cutoff`。区分新闻首发时间和你实际可获得该信息的时间。不要将目标日收盘后新闻或未来标签写入预测该日的输入；模型生成的模拟市场文本应使用 `is_synthetic=true` 并记录生成来源。

- **股票级文本**：按 `(dataset, stock_uid, 时间)` 归属，再用本表映射到 `array_index`；名称和 QID 用于辅助实体识别，不能代替经过核对的连接键。
- **市场级文本**：可用 `entity_scope='market'`、`stock_uid=null` 保存，按时间对齐后再广播或经融合模块供股票使用，不必伪造股票编号。
- **张量对齐**：如果文本已编码为向量，建议保存为 `[N,T,D_text]` 并使用与价格完全相同的股票轴/日期轴，同时另存文本有效性掩码。缺失文本和缺失价格应分开标记。
- **训练窗口**：当前 `lookback=16, steps=1` 时，价格输入为 `offset..offset+15`，标签日期为 `offset+16`；文本截止时间必须与实际预测时点一致，而非默认采用标签日全天新闻。
- **日期轴**：NASDAQ/NYSE 的索引 0 对应 2013-01-02，索引 1244 对应 2017-12-08。SP500 原数组索引 0 对应 2012-05-14，训练时先执行 `[:,915:,:]`，因此训练使用数组索引 0 对应原始索引 915，即 2016-01-04；股票轴不受该截取影响。构造 SP500 文本时也必须做相同日期截取。

Python 读取机器可用映射示例：

```python
import pandas as pd
mapping = pd.read_csv("docs/stock_identity_mapping.csv", keep_default_na=False)
sp500 = mapping[mapping["dataset"] == "SP500"].sort_values("array_index")
assert sp500["array_index"].tolist() == list(range(474))
assert sp500.iloc[0]["ticker"] == "POOL"
# 后续将逐股文本按 stock_uid / 可获得时间连接，再排列成价格数组的轴顺序。
```

## 5. 来源、复现与验证

- 当前数值输入：`dataset/NASDAQ/eod_data.pkl`、`dataset/NYSE/eod_data.pkl`、`dataset/SP500/SP500.npy`。
- NASDAQ/NYSE 股票列表与 QID：`../Temporal_Relational_Stock_Ranking-master/data/*_tickers_qualify_dr-0.98_min-5_smooth.csv`、`*_wiki.csv`；路径相对于 StockMixer 项目根目录。
- SP500 顺序：`dataset/SP500/baseline_data_sp500.npy`；名称：`dataset/SP500/sp500_ticker.csv`。
- 当前实现：`src/load_data.py`、`src/train_strict.py`、`src/model.py`。主干不接收名称或 QID，本文提供的是文本扩展的对齐元数据，不修改训练逻辑或数据数组。
- [机器可读 CSV](stock_identity_mapping.csv)：完整 3237 行，包含名称来源状态、QID 状态、数据日期范围。
- [核验清单与输入文件 SHA256](stock_identity_mapping_validation.json)：用于确认该映射属于哪一版数组。
- 复现：在项目根目录的 stockmixer 环境运行 `python scripts/export_stock_identity_mapping.py`。

后续更换数据文件、股票集合或重排股票轴后，应重新生成并核验本映射；不得沿用旧行号。

## 6. 完整索引—股票映射表

以下 `array_index` 均从 0 开始；“建议 stock_uid”是新增连接键。公司名来自本地 SP500 名称快照，未提供处仍可直接使用 ticker 对齐；“—”表示无可用的源 QID。

### NASDAQ：1026 只

| array_index | 建议 stock_uid | 股票代码 ticker | 公司名称（本地快照） | 源 Wikidata QID |
|---:|---|---|---|---|
| 0 | NASDAQ:AABA | AABA | 未提供 | Q30321616 |
| 1 | NASDAQ:AAON | AAON | 未提供 | Q286214 |
| 2 | NASDAQ:AAPL | AAPL | Apple | Q312 |
| 3 | NASDAQ:AAWW | AAWW | 未提供 | — |
| 4 | NASDAQ:AAXJ | AAXJ | 未提供 | — |
| 5 | NASDAQ:AAXN | AAXN | 未提供 | — |
| 6 | NASDAQ:ABAX | ABAX | 未提供 | — |
| 7 | NASDAQ:ABCB | ABCB | 未提供 | — |
| 8 | NASDAQ:ABCO | ABCO | 未提供 | Q7712716 |
| 9 | NASDAQ:ABMD | ABMD | Abiomed | — |
| 10 | NASDAQ:ACET | ACET | 未提供 | — |
| 11 | NASDAQ:ACGL | ACGL | 未提供 | — |
| 12 | NASDAQ:ACHC | ACHC | 未提供 | — |
| 13 | NASDAQ:ACIW | ACIW | 未提供 | Q4650685 |
| 14 | NASDAQ:ACOR | ACOR | 未提供 | Q4674531 |
| 15 | NASDAQ:ACTA | ACTA | 未提供 | Q5969568 |
| 16 | NASDAQ:ACWI | ACWI | 未提供 | — |
| 17 | NASDAQ:ACWX | ACWX | 未提供 | — |
| 18 | NASDAQ:ACXM | ACXM | 未提供 | Q344935 |
| 19 | NASDAQ:ADBE | ADBE | Adobe | Q11463 |
| 20 | NASDAQ:ADI | ADI | Analog Devices | Q484930 |
| 21 | NASDAQ:ADP | ADP | Automatic Data Processing | Q6394386 |
| 22 | NASDAQ:ADRD | ADRD | 未提供 | — |
| 23 | NASDAQ:ADRE | ADRE | 未提供 | — |
| 24 | NASDAQ:ADSK | ADSK | Autodesk | Q628051 |
| 25 | NASDAQ:ADTN | ADTN | 未提供 | Q4651203 |
| 26 | NASDAQ:ADUS | ADUS | 未提供 | — |
| 27 | NASDAQ:AEGN | AEGN | 未提供 | Q22060267 |
| 28 | NASDAQ:AEIS | AEIS | 未提供 | Q4686194 |
| 29 | NASDAQ:AFAM | AFAM | 未提供 | — |
| 30 | NASDAQ:AFSI | AFSI | 未提供 | Q4738904 |
| 31 | NASDAQ:AGII | AGII | 未提供 | Q17049669 |
| 32 | NASDAQ:AGNC | AGNC | 未提供 | — |
| 33 | NASDAQ:AGYS | AGYS | 未提供 | — |
| 34 | NASDAQ:AHGP | AHGP | 未提供 | — |
| 35 | NASDAQ:AIA | AIA | 未提供 | — |
| 36 | NASDAQ:AIMC | AIMC | 未提供 | — |
| 37 | NASDAQ:AKAM | AKAM | Akamai Technologies | Q415598 |
| 38 | NASDAQ:AKRX | AKRX | 未提供 | — |
| 39 | NASDAQ:ALCO | ALCO | 未提供 | — |
| 40 | NASDAQ:ALGN | ALGN | Align Technology | — |
| 41 | NASDAQ:ALGT | ALGT | 未提供 | — |
| 42 | NASDAQ:ALKS | ALKS | 未提供 | Q4727688 |
| 43 | NASDAQ:ALNY | ALNY | 未提供 | Q16974682 |
| 44 | NASDAQ:ALOG | ALOG | 未提供 | Q4751068 |
| 45 | NASDAQ:ALXN | ALXN | 未提供 | Q204727 |
| 46 | NASDAQ:AMAG | AMAG | 未提供 | Q16974741 |
| 47 | NASDAQ:AMAT | AMAT | Applied Materials | Q7449678 |
| 48 | NASDAQ:AMBA | AMBA | 未提供 | Q4741030 |
| 49 | NASDAQ:AMCX | AMCX | 未提供 | Q4652575 |
| 50 | NASDAQ:AMED | AMED | 未提供 | Q4742102 |
| 51 | NASDAQ:AMGN | AMGN | Amgen | Q470517 |
| 52 | NASDAQ:AMNB | AMNB | 未提供 | — |
| 53 | NASDAQ:AMOT | AMOT | 未提供 | — |
| 54 | NASDAQ:AMSF | AMSF | 未提供 | — |
| 55 | NASDAQ:AMSWA | AMSWA | 未提供 | — |
| 56 | NASDAQ:AMTD | AMTD | 未提供 | — |
| 57 | NASDAQ:AMWD | AMWD | 未提供 | Q4745391 |
| 58 | NASDAQ:AMZN | AMZN | Amazon | Q3884 |
| 59 | NASDAQ:ANAT | ANAT | 未提供 | Q4744566 |
| 60 | NASDAQ:ANCX | ANCX | 未提供 | — |
| 61 | NASDAQ:ANDE | ANDE | 未提供 | Q7713699 |
| 62 | NASDAQ:ANGO | ANGO | 未提供 | — |
| 63 | NASDAQ:ANIK | ANIK | 未提供 | — |
| 64 | NASDAQ:ANSS | ANSS | Ansys | Q295774 |
| 65 | NASDAQ:AOBC | AOBC | 未提供 | — |
| 66 | NASDAQ:AOSL | AOSL | 未提供 | — |
| 67 | NASDAQ:APEI | APEI | 未提供 | — |
| 68 | NASDAQ:APLP | APLP | 未提供 | — |
| 69 | NASDAQ:APOG | APOG | 未提供 | — |
| 70 | NASDAQ:ARCB | ARCB | 未提供 | Q4650296 |
| 71 | NASDAQ:ARCC | ARCC | 未提供 | Q4789079 |
| 72 | NASDAQ:ARII | ARII | 未提供 | — |
| 73 | NASDAQ:ARLP | ARLP | 未提供 | — |
| 74 | NASDAQ:ARNA | ARNA | 未提供 | Q4788901 |
| 75 | NASDAQ:AROW | AROW | 未提供 | — |
| 76 | NASDAQ:ARTNA | ARTNA | 未提供 | — |
| 77 | NASDAQ:ASCMA | ASCMA | 未提供 | — |
| 78 | NASDAQ:ASFI | ASFI | 未提供 | — |
| 79 | NASDAQ:ASML | ASML | 未提供 | Q297879 |
| 80 | NASDAQ:ASPS | ASPS | 未提供 | Q7077162 |
| 81 | NASDAQ:ASTE | ASTE | 未提供 | — |
| 82 | NASDAQ:ATHN | ATHN | 未提供 | Q4813401 |
| 83 | NASDAQ:ATLO | ATLO | 未提供 | — |
| 84 | NASDAQ:ATNI | ATNI | 未提供 | Q4816569 |
| 85 | NASDAQ:ATRC | ATRC | 未提供 | — |
| 86 | NASDAQ:ATRI | ATRI | 未提供 | — |
| 87 | NASDAQ:ATRO | ATRO | 未提供 | — |
| 88 | NASDAQ:ATVI | ATVI | Activision Blizzard | Q344341 |
| 89 | NASDAQ:AVAV | AVAV | 未提供 | Q919300 |
| 90 | NASDAQ:AVHI | AVHI | 未提供 | — |
| 91 | NASDAQ:AVNW | AVNW | 未提供 | Q4828605 |
| 92 | NASDAQ:AZPN | AZPN | 未提供 | — |
| 93 | NASDAQ:BABY | BABY | 未提供 | — |
| 94 | NASDAQ:BANF | BANF | 未提供 | Q4854030 |
| 95 | NASDAQ:BANR | BANR | 未提供 | — |
| 96 | NASDAQ:BBBY | BBBY | 未提供 | Q813782 |
| 97 | NASDAQ:BCPC | BCPC | 未提供 | — |
| 98 | NASDAQ:BDGE | BDGE | 未提供 | — |
| 99 | NASDAQ:BECN | BECN | 未提供 | — |
| 100 | NASDAQ:BELFB | BELFB | 未提供 | — |
| 101 | NASDAQ:BFIN | BFIN | 未提供 | — |
| 102 | NASDAQ:BGFV | BGFV | 未提供 | Q4904902 |
| 103 | NASDAQ:BIB | BIB | 未提供 | — |
| 104 | NASDAQ:BIDU | BIDU | 未提供 | Q14772 |
| 105 | NASDAQ:BIIB | BIIB | Biogen | Q864338 |
| 106 | NASDAQ:BIS | BIS | 未提供 | — |
| 107 | NASDAQ:BJRI | BJRI | 未提供 | Q4835755 |
| 108 | NASDAQ:BKCC | BKCC | 未提供 | Q6386533 |
| 109 | NASDAQ:BLKB | BLKB | 未提供 | Q4922539 |
| 110 | NASDAQ:BLMN | BLMN | 未提供 | Q4928282 |
| 111 | NASDAQ:BLMT | BLMT | 未提供 | — |
| 112 | NASDAQ:BMRC | BMRC | 未提供 | — |
| 113 | NASDAQ:BMRN | BMRN | 未提供 | Q4914632 |
| 114 | NASDAQ:BMTC | BMTC | 未提供 | Q18156655 |
| 115 | NASDAQ:BOBE | BOBE | 未提供 | Q4932386 |
| 116 | NASDAQ:BOFI | BOFI | 未提供 | Q4937270 |
| 117 | NASDAQ:BOKF | BOKF | 未提供 | Q2162158 |
| 118 | NASDAQ:BOOM | BOOM | 未提供 | Q5318916 |
| 119 | NASDAQ:BPFH | BPFH | 未提供 | — |
| 120 | NASDAQ:BPOP | BPOP | 未提供 | Q7229656 |
| 121 | NASDAQ:BPOPN | BPOPN | 未提供 | — |
| 122 | NASDAQ:BRCD | BRCD | 未提供 | Q2424396 |
| 123 | NASDAQ:BREW | BREW | 未提供 | Q7998742 |
| 124 | NASDAQ:BRKL | BRKL | 未提供 | — |
| 125 | NASDAQ:BRKR | BRKR | 未提供 | Q431454 |
| 126 | NASDAQ:BRKS | BRKS | 未提供 | — |
| 127 | NASDAQ:BSET | BSET | 未提供 | Q4868109 |
| 128 | NASDAQ:BSF | BSF | 未提供 | — |
| 129 | NASDAQ:BSFT | BSFT | 未提供 | — |
| 130 | NASDAQ:BSRR | BSRR | 未提供 | — |
| 131 | NASDAQ:BSTC | BSTC | 未提供 | — |
| 132 | NASDAQ:BUSE | BUSE | 未提供 | Q5001347 |
| 133 | NASDAQ:BWINB | BWINB | 未提供 | — |
| 134 | NASDAQ:BWLD | BWLD | 未提供 | Q509255 |
| 135 | NASDAQ:CA | CA | 未提供 | Q600624 |
| 136 | NASDAQ:CAC | CAC | 未提供 | — |
| 137 | NASDAQ:CACC | CACC | 未提供 | Q15845093 |
| 138 | NASDAQ:CAKE | CAKE | 未提供 | Q1045842 |
| 139 | NASDAQ:CALL | CALL | 未提供 | Q7939183 |
| 140 | NASDAQ:CALM | CALM | 未提供 | Q5018092 |
| 141 | NASDAQ:CAMP | CAMP | 未提供 | — |
| 142 | NASDAQ:CAR | CAR | 未提供 | Q2874395 |
| 143 | NASDAQ:CARB | CARB | 未提供 | Q5038056 |
| 144 | NASDAQ:CARZ | CARZ | 未提供 | — |
| 145 | NASDAQ:CASH | CASH | 未提供 | Q6822267 |
| 146 | NASDAQ:CASS | CASS | 未提供 | — |
| 147 | NASDAQ:CASY | CASY | 未提供 | Q2940968 |
| 148 | NASDAQ:CATY | CATY | 未提供 | Q4189507 |
| 149 | NASDAQ:CAVM | CAVM | 未提供 | Q5055187 |
| 150 | NASDAQ:CBF | CBF | 未提供 | Q5035481 |
| 151 | NASDAQ:CBOE | CBOE | Cboe Global Markets | Q1071853 |
| 152 | NASDAQ:CBRL | CBRL | 未提供 | Q4492609 |
| 153 | NASDAQ:CBSH | CBSH | 未提供 | Q5152411 |
| 154 | NASDAQ:CCBG | CCBG | 未提供 | — |
| 155 | NASDAQ:CCMP | CCMP | 未提供 | — |
| 156 | NASDAQ:CCNE | CCNE | 未提供 | — |
| 157 | NASDAQ:CCOI | CCOI | 未提供 | Q1106938 |
| 158 | NASDAQ:CDNS | CDNS | Cadence Design Systems | Q608776 |
| 159 | NASDAQ:CELG | CELG | 未提供 | Q842947 |
| 160 | NASDAQ:CENT | CENT | 未提供 | — |
| 161 | NASDAQ:CENTA | CENTA | 未提供 | — |
| 162 | NASDAQ:CERN | CERN | Cerner | Q1055390 |
| 163 | NASDAQ:CEVA | CEVA | 未提供 | — |
| 164 | NASDAQ:CFFI | CFFI | 未提供 | — |
| 165 | NASDAQ:CG | CG | 未提供 | Q926806 |
| 166 | NASDAQ:CGNX | CGNX | 未提供 | Q2981857 |
| 167 | NASDAQ:CGO | CGO | 未提供 | — |
| 168 | NASDAQ:CHCO | CHCO | 未提供 | — |
| 169 | NASDAQ:CHDN | CHDN | 未提供 | Q5118174 |
| 170 | NASDAQ:CHEF | CHEF | 未提供 | — |
| 171 | NASDAQ:CHFC | CHFC | 未提供 | Q5090406 |
| 172 | NASDAQ:CHI | CHI | 未提供 | — |
| 173 | NASDAQ:CHKP | CHKP | 未提供 | Q176587 |
| 174 | NASDAQ:CHRW | CHRW | C. H. Robinson | Q931200 |
| 175 | NASDAQ:CHSCP | CHSCP | 未提供 | Q5011381 |
| 176 | NASDAQ:CHUY | CHUY | 未提供 | Q5118415 |
| 177 | NASDAQ:CHW | CHW | 未提供 | — |
| 178 | NASDAQ:CHY | CHY | 未提供 | — |
| 179 | NASDAQ:CIGI | CIGI | 未提供 | Q5147194 |
| 180 | NASDAQ:CINF | CINF | Cincinnati Financial | Q5120253 |
| 181 | NASDAQ:CIU | CIU | 未提供 | — |
| 182 | NASDAQ:CIVB | CIVB | 未提供 | — |
| 183 | NASDAQ:CLCT | CLCT | 未提供 | — |
| 184 | NASDAQ:CMCO | CMCO | 未提供 | — |
| 185 | NASDAQ:CMCSA | CMCSA | Comcast | Q1113804 |
| 186 | NASDAQ:CME | CME | CME Group | Q1023876 |
| 187 | NASDAQ:CMPR | CMPR | 未提供 | Q2128566 |
| 188 | NASDAQ:CMTL | CMTL | 未提供 | — |
| 189 | NASDAQ:CNBKA | CNBKA | 未提供 | — |
| 190 | NASDAQ:CNMD | CNMD | 未提供 | — |
| 191 | NASDAQ:CNOB | CNOB | 未提供 | — |
| 192 | NASDAQ:CNSL | CNSL | 未提供 | — |
| 193 | NASDAQ:CNXN | CNXN | 未提供 | — |
| 194 | NASDAQ:COBZ | COBZ | 未提供 | — |
| 195 | NASDAQ:COHR | COHR | 未提供 | Q1107088 |
| 196 | NASDAQ:COHU | COHU | 未提供 | — |
| 197 | NASDAQ:COKE | COKE | 未提供 | Q5139058 |
| 198 | NASDAQ:COLB | COLB | 未提供 | — |
| 199 | NASDAQ:COLM | COLM | 未提供 | Q1112588 |
| 200 | NASDAQ:CONN | CONN | 未提供 | Q5161257 |
| 201 | NASDAQ:CORE | CORE | 未提供 | Q5170128 |
| 202 | NASDAQ:COST | COST | Costco | Q715583 |
| 203 | NASDAQ:CPLA | CPLA | 未提供 | Q5035257 |
| 204 | NASDAQ:CPSI | CPSI | 未提供 | — |
| 205 | NASDAQ:CRAI | CRAI | 未提供 | Q5013834 |
| 206 | NASDAQ:CRAY | CRAY | 未提供 | Q144060 |
| 207 | NASDAQ:CRED | CRED | 未提供 | — |
| 208 | NASDAQ:CREE | CREE | 未提供 | Q5183828 |
| 209 | NASDAQ:CRESY | CRESY | 未提供 | — |
| 210 | NASDAQ:CRMT | CRMT | 未提供 | — |
| 211 | NASDAQ:CROX | CROX | 未提供 | Q926699 |
| 212 | NASDAQ:CRUS | CRUS | 未提供 | Q1067998 |
| 213 | NASDAQ:CRVL | CRVL | 未提供 | Q5169368 |
| 214 | NASDAQ:CRZO | CRZO | 未提供 | — |
| 215 | NASDAQ:CSCO | CSCO | Cisco Systems | Q173395 |
| 216 | NASDAQ:CSFL | CSFL | 未提供 | — |
| 217 | NASDAQ:CSGP | CSGP | 未提供 | Q16988748 |
| 218 | NASDAQ:CSGS | CSGS | 未提供 | — |
| 219 | NASDAQ:CSII | CSII | 未提供 | — |
| 220 | NASDAQ:CSJ | CSJ | 未提供 | — |
| 221 | NASDAQ:CSOD | CSOD | 未提供 | — |
| 222 | NASDAQ:CSPI | CSPI | 未提供 | — |
| 223 | NASDAQ:CSQ | CSQ | 未提供 | — |
| 224 | NASDAQ:CSTE | CSTE | 未提供 | Q1287771 |
| 225 | NASDAQ:CSX | CSX | CSX | — |
| 226 | NASDAQ:CTAS | CTAS | Cintas Corporation | Q15109928 |
| 227 | NASDAQ:CTBI | CTBI | 未提供 | — |
| 228 | NASDAQ:CTRN | CTRN | 未提供 | Q5122438 |
| 229 | NASDAQ:CTSH | CTSH | Cognizant Technology Solutions | Q1107035 |
| 230 | NASDAQ:CTWS | CTWS | 未提供 | — |
| 231 | NASDAQ:CTXS | CTXS | Citrix Systems | Q916196 |
| 232 | NASDAQ:CUTR | CUTR | 未提供 | — |
| 233 | NASDAQ:CVBF | CVBF | 未提供 | — |
| 234 | NASDAQ:CVCO | CVCO | 未提供 | — |
| 235 | NASDAQ:CVCY | CVCY | 未提供 | — |
| 236 | NASDAQ:CVGW | CVGW | 未提供 | Q16850385 |
| 237 | NASDAQ:CVLT | CVLT | 未提供 | — |
| 238 | NASDAQ:CVV | CVV | 未提供 | — |
| 239 | NASDAQ:CWCO | CWCO | 未提供 | — |
| 240 | NASDAQ:CY | CY | 未提供 | Q478354 |
| 241 | NASDAQ:CYOU | CYOU | 未提供 | Q5072155 |
| 242 | NASDAQ:CZNC | CZNC | 未提供 | — |
| 243 | NASDAQ:DAKT | DAKT | 未提供 | Q1157767 |
| 244 | NASDAQ:DCOM | DCOM | 未提供 | Q5277215 |
| 245 | NASDAQ:DFRG | DFRG | 未提供 | Q21070725 |
| 246 | NASDAQ:DGICA | DGICA | 未提供 | — |
| 247 | NASDAQ:DGII | DGII | 未提供 | Q8041701 |
| 248 | NASDAQ:DGLD | DGLD | 未提供 | — |
| 249 | NASDAQ:DHIL | DHIL | 未提供 | — |
| 250 | NASDAQ:DIOD | DIOD | 未提供 | Q5279096 |
| 251 | NASDAQ:DISCK | DISCK | Discovery (Series C) | — |
| 252 | NASDAQ:DISH | DISH | Dish Network | Q1199757 |
| 253 | NASDAQ:DLBS | DLBS | 未提供 | — |
| 254 | NASDAQ:DLTR | DLTR | Dollar Tree | Q5289230 |
| 255 | NASDAQ:DMLP | DMLP | 未提供 | — |
| 256 | NASDAQ:DMRC | DMRC | 未提供 | Q5275736 |
| 257 | NASDAQ:DNKN | DNKN | 未提供 | Q847743 |
| 258 | NASDAQ:DORM | DORM | 未提供 | — |
| 259 | NASDAQ:DSGX | DSGX | 未提供 | — |
| 260 | NASDAQ:DSLV | DSLV | 未提供 | — |
| 261 | NASDAQ:DSPG | DSPG | 未提供 | Q5206170 |
| 262 | NASDAQ:DTYS | DTYS | 未提供 | — |
| 263 | NASDAQ:DVY | DVY | 未提供 | — |
| 264 | NASDAQ:DWAQ | DWAQ | 未提供 | — |
| 265 | NASDAQ:DWAS | DWAS | 未提供 | — |
| 266 | NASDAQ:DWPP | DWPP | 未提供 | — |
| 267 | NASDAQ:DXCM | DXCM | DexCom | Q15109865 |
| 268 | NASDAQ:DXPE | DXPE | 未提供 | — |
| 269 | NASDAQ:EA | EA | Electronic Arts | Q173941 |
| 270 | NASDAQ:EBTC | EBTC | 未提供 | — |
| 271 | NASDAQ:ECHO | ECHO | 未提供 | Q5332702 |
| 272 | NASDAQ:ECOL | ECOL | 未提供 | — |
| 273 | NASDAQ:ECPG | ECPG | 未提供 | Q5375543 |
| 274 | NASDAQ:EEFT | EEFT | 未提供 | Q5412010 |
| 275 | NASDAQ:EEMA | EEMA | 未提供 | — |
| 276 | NASDAQ:EFII | EFII | 未提供 | Q5358494 |
| 277 | NASDAQ:EFSC | EFSC | 未提供 | — |
| 278 | NASDAQ:EGBN | EGBN | 未提供 | Q5324991 |
| 279 | NASDAQ:EGOV | EGOV | 未提供 | Q17114312 |
| 280 | NASDAQ:EMB | EMB | 未提供 | — |
| 281 | NASDAQ:EMCB | EMCB | 未提供 | — |
| 282 | NASDAQ:EMCI | EMCI | 未提供 | — |
| 283 | NASDAQ:EMIF | EMIF | 未提供 | — |
| 284 | NASDAQ:EML | EML | 未提供 | — |
| 285 | NASDAQ:ENSG | ENSG | 未提供 | — |
| 286 | NASDAQ:ENTG | ENTG | 未提供 | Q5380128 |
| 287 | NASDAQ:ENZL | ENZL | 未提供 | — |
| 288 | NASDAQ:EPAY | EPAY | 未提供 | — |
| 289 | NASDAQ:ERIE | ERIE | 未提供 | Q5388314 |
| 290 | NASDAQ:ESCA | ESCA | 未提供 | — |
| 291 | NASDAQ:ESGR | ESGR | 未提供 | — |
| 292 | NASDAQ:ESLT | ESLT | 未提供 | Q1325369 |
| 293 | NASDAQ:ESND | ESND | 未提供 | — |
| 294 | NASDAQ:ESRX | ESRX | 未提供 | Q768231 |
| 295 | NASDAQ:ESSA | ESSA | 未提供 | — |
| 296 | NASDAQ:ETFC | ETFC | 未提供 | Q2607502 |
| 297 | NASDAQ:EUFN | EUFN | 未提供 | — |
| 298 | NASDAQ:EWBC | EWBC | 未提供 | Q5329659 |
| 299 | NASDAQ:EWZS | EWZS | 未提供 | — |
| 300 | NASDAQ:EXA | EXA | 未提供 | Q3061734 |
| 301 | NASDAQ:EXAC | EXAC | 未提供 | — |
| 302 | NASDAQ:EXLS | EXLS | 未提供 | Q5324689 |
| 303 | NASDAQ:EXPD | EXPD | Expeditors | Q1384324 |
| 304 | NASDAQ:EXPE | EXPE | Expedia Group | Q8085893 |
| 305 | NASDAQ:EXPO | EXPO | 未提供 | Q5421507 |
| 306 | NASDAQ:FANG | FANG | Diamondback Energy | — |
| 307 | NASDAQ:FARM | FARM | 未提供 | Q18148458 |
| 308 | NASDAQ:FARO | FARO | 未提供 | — |
| 309 | NASDAQ:FAST | FAST | Fastenal | Q3067119 |
| 310 | NASDAQ:FB | FB | Facebook | Q380 |
| 311 | NASDAQ:FBNC | FBNC | 未提供 | Q5452332 |
| 312 | NASDAQ:FBNK | FBNK | 未提供 | — |
| 313 | NASDAQ:FCBC | FCBC | 未提供 | Q5452775 |
| 314 | NASDAQ:FCNCA | FCNCA | 未提供 | Q5452733 |
| 315 | NASDAQ:FDEF | FDEF | 未提供 | Q5452995 |
| 316 | NASDAQ:FDUS | FDUS | 未提供 | — |
| 317 | NASDAQ:FELE | FELE | 未提供 | — |
| 318 | NASDAQ:FFBC | FFBC | 未提供 | Q5453009 |
| 319 | NASDAQ:FFIC | FFIC | 未提供 | — |
| 320 | NASDAQ:FFIN | FFIN | 未提供 | — |
| 321 | NASDAQ:FFIV | FFIV | F5 Networks | Q3029296 |
| 322 | NASDAQ:FFKT | FFKT | 未提供 | — |
| 323 | NASDAQ:FFNW | FFNW | 未提供 | — |
| 324 | NASDAQ:FIBK | FIBK | 未提供 | Q5453107 |
| 325 | NASDAQ:FINL | FINL | 未提供 | Q5450341 |
| 326 | NASDAQ:FISI | FISI | 未提供 | — |
| 327 | NASDAQ:FISV | FISV | Fiserv | Q1420513 |
| 328 | NASDAQ:FITB | FITB | Fifth Third Bancorp | Q1411810 |
| 329 | NASDAQ:FIVE | FIVE | 未提供 | Q5455836 |
| 330 | NASDAQ:FIZZ | FIZZ | 未提供 | Q15109910 |
| 331 | NASDAQ:FLEX | FLEX | 未提供 | Q1140680 |
| 332 | NASDAQ:FLIC | FLIC | 未提供 | — |
| 333 | NASDAQ:FLIR | FLIR | 未提供 | Q5426537 |
| 334 | NASDAQ:FLXS | FLXS | 未提供 | — |
| 335 | NASDAQ:FMBI | FMBI | 未提供 | Q5453330 |
| 336 | NASDAQ:FMNB | FMNB | 未提供 | — |
| 337 | NASDAQ:FNGN | FNGN | 未提供 | — |
| 338 | NASDAQ:FNHC | FNHC | 未提供 | — |
| 339 | NASDAQ:FNLC | FNLC | 未提供 | — |
| 340 | NASDAQ:FNSR | FNSR | 未提供 | — |
| 341 | NASDAQ:FORR | FORR | 未提供 | Q829961 |
| 342 | NASDAQ:FOSL | FOSL | 未提供 | Q356212 |
| 343 | NASDAQ:FOX | FOX | Fox Corporation (Class B) | — |
| 344 | NASDAQ:FOXA | FOXA | Fox Corporation (Class A) | Q5476713 |
| 345 | NASDAQ:FRAN | FRAN | 未提供 | — |
| 346 | NASDAQ:FRGI | FRGI | 未提供 | Q16976069 |
| 347 | NASDAQ:FRME | FRME | 未提供 | Q5453261 |
| 348 | NASDAQ:FRPH | FRPH | 未提供 | Q18636301 |
| 349 | NASDAQ:FSLR | FSLR | 未提供 | Q1138888 |
| 350 | NASDAQ:FSTR | FSTR | 未提供 | — |
| 351 | NASDAQ:FTCS | FTCS | 未提供 | — |
| 352 | NASDAQ:FTNT | FTNT | Fortinet | Q2749364 |
| 353 | NASDAQ:FTR | FTR | 未提供 | Q2974454 |
| 354 | NASDAQ:FTRI | FTRI | 未提供 | — |
| 355 | NASDAQ:FULT | FULT | 未提供 | Q5508431 |
| 356 | NASDAQ:FUNC | FUNC | 未提供 | — |
| 357 | NASDAQ:FWRD | FWRD | 未提供 | — |
| 358 | NASDAQ:GABC | GABC | 未提供 | — |
| 359 | NASDAQ:GAIN | GAIN | 未提供 | — |
| 360 | NASDAQ:GBCI | GBCI | 未提供 | Q17183462 |
| 361 | NASDAQ:GBDC | GBDC | 未提供 | — |
| 362 | NASDAQ:GBNK | GBNK | 未提供 | — |
| 363 | NASDAQ:GDEN | GDEN | 未提供 | Q5579415 |
| 364 | NASDAQ:GEOS | GEOS | 未提供 | — |
| 365 | NASDAQ:GGAL | GGAL | 未提供 | Q3118234 |
| 366 | NASDAQ:GHDX | GHDX | 未提供 | Q28175069 |
| 367 | NASDAQ:GIFI | GIFI | 未提供 | — |
| 368 | NASDAQ:GIII | GIII | 未提供 | — |
| 369 | NASDAQ:GILD | GILD | Gilead Sciences | Q663596 |
| 370 | NASDAQ:GLNG | GLNG | 未提供 | Q5578424 |
| 371 | NASDAQ:GLRE | GLRE | 未提供 | — |
| 372 | NASDAQ:GMLP | GMLP | 未提供 | — |
| 373 | NASDAQ:GNCMA | GNCMA | 未提供 | Q5512990 |
| 374 | NASDAQ:GNMA | GNMA | 未提供 | — |
| 375 | NASDAQ:GOLD | GOLD | 未提供 | Q1961738 |
| 376 | NASDAQ:GOOD | GOOD | 未提供 | — |
| 377 | NASDAQ:GOOG | GOOG | Alphabet (Class C) | Q95 |
| 378 | NASDAQ:GOOGL | GOOGL | Alphabet (Class A) | Q20800404 |
| 379 | NASDAQ:GOV | GOV | 未提供 | — |
| 380 | NASDAQ:GPOR | GPOR | 未提供 | — |
| 381 | NASDAQ:GPRE | GPRE | 未提供 | — |
| 382 | NASDAQ:GRFS | GRFS | 未提供 | Q901262 |
| 383 | NASDAQ:GRMN | GRMN | Garmin | Q658789 |
| 384 | NASDAQ:GSBC | GSBC | 未提供 | — |
| 385 | NASDAQ:GT | GT | 未提供 | Q620875 |
| 386 | NASDAQ:GTLS | GTLS | 未提供 | — |
| 387 | NASDAQ:GULF | GULF | 未提供 | — |
| 388 | NASDAQ:HA | HA | 未提供 | Q642610 |
| 389 | NASDAQ:HAFC | HAFC | 未提供 | Q5648633 |
| 390 | NASDAQ:HAIN | HAIN | 未提供 | Q5639346 |
| 391 | NASDAQ:HALL | HALL | 未提供 | — |
| 392 | NASDAQ:HALO | HALO | 未提供 | Q5643474 |
| 393 | NASDAQ:HAS | HAS | Hasbro | Q501476 |
| 394 | NASDAQ:HAYN | HAYN | 未提供 | Q1591976 |
| 395 | NASDAQ:HBAN | HBAN | Huntington Bancshares | Q798819 |
| 396 | NASDAQ:HBCP | HBCP | 未提供 | — |
| 397 | NASDAQ:HBHC | HBHC | 未提供 | Q5647025 |
| 398 | NASDAQ:HBNC | HBNC | 未提供 | — |
| 399 | NASDAQ:HCCI | HCCI | 未提供 | — |
| 400 | NASDAQ:HCOM | HCOM | 未提供 | Q5684575 |
| 401 | NASDAQ:HCSG | HCSG | 未提供 | — |
| 402 | NASDAQ:HDNG | HDNG | 未提供 | Q5656147 |
| 403 | NASDAQ:HEES | HEES | 未提供 | — |
| 404 | NASDAQ:HELE | HELE | 未提供 | Q5703474 |
| 405 | NASDAQ:HFBC | HFBC | 未提供 | — |
| 406 | NASDAQ:HFWA | HFWA | 未提供 | — |
| 407 | NASDAQ:HIBB | HIBB | 未提供 | Q5750671 |
| 408 | NASDAQ:HMST | HMST | 未提供 | — |
| 409 | NASDAQ:HMSY | HMSY | 未提供 | — |
| 410 | NASDAQ:HOFT | HOFT | 未提供 | — |
| 411 | NASDAQ:HOLI | HOLI | 未提供 | — |
| 412 | NASDAQ:HOLX | HOLX | Hologic | Q5884072 |
| 413 | NASDAQ:HOMB | HOMB | 未提供 | — |
| 414 | NASDAQ:HOPE | HOPE | 未提供 | — |
| 415 | NASDAQ:HPT | HPT | 未提供 | — |
| 416 | NASDAQ:HQCL | HQCL | 未提供 | Q265176 |
| 417 | NASDAQ:HRZN | HRZN | 未提供 | — |
| 418 | NASDAQ:HSIC | HSIC | Henry Schein | Q905086 |
| 419 | NASDAQ:HSII | HSII | 未提供 | Q17110635 |
| 420 | NASDAQ:HSKA | HSKA | 未提供 | — |
| 421 | NASDAQ:HSNI | HSNI | 未提供 | Q452340 |
| 422 | NASDAQ:HSTM | HSTM | 未提供 | — |
| 423 | NASDAQ:HTBI | HTBI | 未提供 | Q16995776 |
| 424 | NASDAQ:HTBK | HTBK | 未提供 | — |
| 425 | NASDAQ:HTHT | HTHT | 未提供 | — |
| 426 | NASDAQ:HTLD | HTLD | 未提供 | — |
| 427 | NASDAQ:HTLF | HTLF | 未提供 | — |
| 428 | NASDAQ:HUBG | HUBG | 未提供 | Q17183768 |
| 429 | NASDAQ:HURC | HURC | 未提供 | — |
| 430 | NASDAQ:HURN | HURN | 未提供 | Q17086972 |
| 431 | NASDAQ:HWKN | HWKN | 未提供 | — |
| 432 | NASDAQ:IAC | IAC | 未提供 | Q974846 |
| 433 | NASDAQ:IART | IART | 未提供 | — |
| 434 | NASDAQ:IBB | IBB | 未提供 | — |
| 435 | NASDAQ:IBKC | IBKC | 未提供 | Q5968523 |
| 436 | NASDAQ:IBKR | IBKR | 未提供 | Q20011294 |
| 437 | NASDAQ:IBOC | IBOC | 未提供 | Q6048685 |
| 438 | NASDAQ:ICFI | ICFI | 未提供 | Q5969562 |
| 439 | NASDAQ:ICLN | ICLN | 未提供 | — |
| 440 | NASDAQ:ICUI | ICUI | 未提供 | Q5969799 |
| 441 | NASDAQ:IDCC | IDCC | 未提供 | Q6044894 |
| 442 | NASDAQ:IDTI | IDTI | 未提供 | Q1665429 |
| 443 | NASDAQ:IDXX | IDXX | Idexx Laboratories | Q1758392 |
| 444 | NASDAQ:IEF | IEF | 未提供 | — |
| 445 | NASDAQ:IEI | IEI | 未提供 | — |
| 446 | NASDAQ:IEP | IEP | 未提供 | Q5985017 |
| 447 | NASDAQ:IFEU | IFEU | 未提供 | — |
| 448 | NASDAQ:IFGL | IFGL | 未提供 | — |
| 449 | NASDAQ:IGF | IGF | 未提供 | — |
| 450 | NASDAQ:IGOV | IGOV | 未提供 | — |
| 451 | NASDAQ:IIIN | IIIN | 未提供 | — |
| 452 | NASDAQ:IIVI | IIVI | 未提供 | — |
| 453 | NASDAQ:IJT | IJT | 未提供 | — |
| 454 | NASDAQ:ILG | ILG | 未提供 | — |
| 455 | NASDAQ:ILMN | ILMN | Illumina | Q2068984 |
| 456 | NASDAQ:IMKTA | IMKTA | 未提供 | Q6032595 |
| 457 | NASDAQ:IMMR | IMMR | 未提供 | — |
| 458 | NASDAQ:IMPV | IMPV | 未提供 | Q6007092 |
| 459 | NASDAQ:INBK | INBK | 未提供 | Q5453106 |
| 460 | NASDAQ:INCY | INCY | Incyte | Q6015591 |
| 461 | NASDAQ:INDB | INDB | 未提供 | Q7355463 |
| 462 | NASDAQ:INDY | INDY | 未提供 | — |
| 463 | NASDAQ:INFN | INFN | 未提供 | Q6029685 |
| 464 | NASDAQ:INTC | INTC | Intel | Q248 |
| 465 | NASDAQ:INTL | INTL | 未提供 | Q1666331 |
| 466 | NASDAQ:INTU | INTU | Intuit | Q1318848 |
| 467 | NASDAQ:INVA | INVA | 未提供 | — |
| 468 | NASDAQ:INWK | INWK | 未提供 | Q6035325 |
| 469 | NASDAQ:IONS | IONS | 未提供 | — |
| 470 | NASDAQ:IOSP | IOSP | 未提供 | Q15109891 |
| 471 | NASDAQ:IPAR | IPAR | 未提供 | — |
| 472 | NASDAQ:IPCC | IPCC | 未提供 | Q6030036 |
| 473 | NASDAQ:IPGP | IPGP | IPG Photonics | Q5973024 |
| 474 | NASDAQ:IPHS | IPHS | 未提供 | — |
| 475 | NASDAQ:IPXL | IPXL | 未提供 | Q16996714 |
| 476 | NASDAQ:IRBT | IRBT | 未提供 | Q285161 |
| 477 | NASDAQ:IRDM | IRDM | 未提供 | Q154387 |
| 478 | NASDAQ:IRWD | IRWD | 未提供 | — |
| 479 | NASDAQ:ISCA | ISCA | 未提供 | Q974211 |
| 480 | NASDAQ:ISHG | ISHG | 未提供 | — |
| 481 | NASDAQ:ISRG | ISRG | Intuitive Surgical | Q2743863 |
| 482 | NASDAQ:ISTB | ISTB | 未提供 | — |
| 483 | NASDAQ:ITRI | ITRI | 未提供 | Q5923178 |
| 484 | NASDAQ:ITRN | ITRN | 未提供 | Q16974955 |
| 485 | NASDAQ:IUSG | IUSG | 未提供 | — |
| 486 | NASDAQ:IUSV | IUSV | 未提供 | — |
| 487 | NASDAQ:IXUS | IXUS | 未提供 | — |
| 488 | NASDAQ:IXYS | IXYS | 未提供 | — |
| 489 | NASDAQ:JACK | JACK | 未提供 | Q1538507 |
| 490 | NASDAQ:JAZZ | JAZZ | 未提供 | Q16255113 |
| 491 | NASDAQ:JBHT | JBHT | J. B. Hunt | Q4041705 |
| 492 | NASDAQ:JBLU | JBLU | 未提供 | Q161086 |
| 493 | NASDAQ:JBSS | JBSS | 未提供 | — |
| 494 | NASDAQ:JCOM | JCOM | 未提供 | Q16997523 |
| 495 | NASDAQ:JJSF | JJSF | 未提供 | Q6109309 |
| 496 | NASDAQ:JKHY | JKHY | Jack Henry & Associates | Q6113060 |
| 497 | NASDAQ:JKI | JKI | 未提供 | — |
| 498 | NASDAQ:JMBA | JMBA | 未提供 | Q3088784 |
| 499 | NASDAQ:JOBS | JOBS | 未提供 | — |
| 500 | NASDAQ:JOUT | JOUT | 未提供 | Q19654979 |
| 501 | NASDAQ:KALU | KALU | 未提供 | Q6348410 |
| 502 | NASDAQ:KBAL | KBAL | 未提供 | Q1526456 |
| 503 | NASDAQ:KBWB | KBWB | 未提供 | — |
| 504 | NASDAQ:KBWD | KBWD | 未提供 | — |
| 505 | NASDAQ:KBWP | KBWP | 未提供 | — |
| 506 | NASDAQ:KBWR | KBWR | 未提供 | — |
| 507 | NASDAQ:KBWY | KBWY | 未提供 | — |
| 508 | NASDAQ:KELYA | KELYA | 未提供 | Q1538585 |
| 509 | NASDAQ:KFRC | KFRC | 未提供 | Q6398722 |
| 510 | NASDAQ:KIRK | KIRK | 未提供 | Q6415714 |
| 511 | NASDAQ:KLAC | KLAC | KLA Corporation | Q624649 |
| 512 | NASDAQ:KLIC | KLIC | 未提供 | — |
| 513 | NASDAQ:KTCC | KTCC | 未提供 | Q1740323 |
| 514 | NASDAQ:KVHI | KVHI | 未提供 | — |
| 515 | NASDAQ:LABL | LABL | 未提供 | — |
| 516 | NASDAQ:LANC | LANC | 未提供 | Q7668516 |
| 517 | NASDAQ:LAWS | LAWS | 未提供 | — |
| 518 | NASDAQ:LBAI | LBAI | 未提供 | — |
| 519 | NASDAQ:LCNB | LCNB | 未提供 | — |
| 520 | NASDAQ:LCUT | LCUT | 未提供 | — |
| 521 | NASDAQ:LECO | LECO | 未提供 | Q6550657 |
| 522 | NASDAQ:LFUS | LFUS | 未提供 | Q6648755 |
| 523 | NASDAQ:LGND | LGND | 未提供 | — |
| 524 | NASDAQ:LHCG | LHCG | 未提供 | — |
| 525 | NASDAQ:LION | LION | 未提供 | Q16250432 |
| 526 | NASDAQ:LKFN | LKFN | 未提供 | — |
| 527 | NASDAQ:LKQ | LKQ | LKQ Corporation | Q20711490 |
| 528 | NASDAQ:LMAT | LMAT | 未提供 | — |
| 529 | NASDAQ:LMNR | LMNR | 未提供 | — |
| 530 | NASDAQ:LMNX | LMNX | 未提供 | — |
| 531 | NASDAQ:LMOS | LMOS | 未提供 | Q6703383 |
| 532 | NASDAQ:LNCE | LNCE | 未提供 | Q7548877 |
| 533 | NASDAQ:LNDC | LNDC | 未提供 | — |
| 534 | NASDAQ:LOGI | LOGI | 未提供 | Q223127 |
| 535 | NASDAQ:LOGM | LOGM | 未提供 | Q5173965 |
| 536 | NASDAQ:LOPE | LOPE | 未提供 | Q33129672 |
| 537 | NASDAQ:LORL | LORL | 未提供 | Q3259355 |
| 538 | NASDAQ:LPLA | LPLA | 未提供 | Q6459734 |
| 539 | NASDAQ:LPNT | LPNT | 未提供 | Q4744658 |
| 540 | NASDAQ:LRCX | LRCX | Lam Research | Q1342041 |
| 541 | NASDAQ:LSTR | LSTR | 未提供 | Q6485221 |
| 542 | NASDAQ:LTXB | LTXB | 未提供 | — |
| 543 | NASDAQ:LULU | LULU | 未提供 | Q6702957 |
| 544 | NASDAQ:LWAY | LWAY | 未提供 | Q6545549 |
| 545 | NASDAQ:LXRX | LXRX | 未提供 | Q6537727 |
| 546 | NASDAQ:LYTS | LYTS | 未提供 | — |
| 547 | NASDAQ:MACK | MACK | 未提供 | — |
| 548 | NASDAQ:MANH | MANH | 未提供 | Q3285819 |
| 549 | NASDAQ:MANT | MANT | 未提供 | Q6746136 |
| 550 | NASDAQ:MAR | MAR | Marriott International | Q1141173 |
| 551 | NASDAQ:MASI | MASI | 未提供 | Q6783278 |
| 552 | NASDAQ:MAT | MAT | 未提供 | Q596139 |
| 553 | NASDAQ:MATW | MATW | 未提供 | — |
| 554 | NASDAQ:MBB | MBB | 未提供 | — |
| 555 | NASDAQ:MBFI | MBFI | 未提供 | — |
| 556 | NASDAQ:MBWM | MBWM | 未提供 | — |
| 557 | NASDAQ:MCHI | MCHI | 未提供 | — |
| 558 | NASDAQ:MCHP | MCHP | Microchip Technology | Q1545076 |
| 559 | NASDAQ:MCRI | MCRI | 未提供 | Q4816705 |
| 560 | NASDAQ:MDCO | MDCO | 未提供 | — |
| 561 | NASDAQ:MDIV | MDIV | 未提供 | — |
| 562 | NASDAQ:MDLZ | MDLZ | Mondelez International | Q12857502 |
| 563 | NASDAQ:MDRX | MDRX | 未提供 | Q4733274 |
| 564 | NASDAQ:MDSO | MDSO | 未提供 | Q6806791 |
| 565 | NASDAQ:MELI | MELI | 未提供 | Q950419 |
| 566 | NASDAQ:MEOH | MEOH | 未提供 | Q13529431 |
| 567 | NASDAQ:MERC | MERC | 未提供 | Q1391599 |
| 568 | NASDAQ:MGEE | MGEE | 未提供 | Q6715705 |
| 569 | NASDAQ:MGLN | MGLN | 未提供 | Q20047296 |
| 570 | NASDAQ:MGRC | MGRC | 未提供 | — |
| 571 | NASDAQ:MHLD | MHLD | 未提供 | Q6735371 |
| 572 | NASDAQ:MINI | MINI | 未提供 | — |
| 573 | NASDAQ:MKSI | MKSI | 未提供 | — |
| 574 | NASDAQ:MKTX | MKTX | MarketAxess | — |
| 575 | NASDAQ:MLAB | MLAB | 未提供 | — |
| 576 | NASDAQ:MLCO | MLCO | 未提供 | — |
| 577 | NASDAQ:MLHR | MLHR | 未提供 | Q489469 |
| 578 | NASDAQ:MLNX | MLNX | 未提供 | Q7680886 |
| 579 | NASDAQ:MMLP | MMLP | 未提供 | — |
| 580 | NASDAQ:MMSI | MMSI | 未提供 | — |
| 581 | NASDAQ:MMYT | MMYT | 未提供 | Q13115286 |
| 582 | NASDAQ:MNRO | MNRO | 未提供 | Q6902090 |
| 583 | NASDAQ:MNTA | MNTA | 未提供 | — |
| 584 | NASDAQ:MOFG | MOFG | 未提供 | Q6843476 |
| 585 | NASDAQ:MORN | MORN | 未提供 | Q1389504 |
| 586 | NASDAQ:MPAA | MPAA | 未提供 | — |
| 587 | NASDAQ:MPWR | MPWR | Monolithic Power Systems | — |
| 588 | NASDAQ:MRCC | MRCC | 未提供 | — |
| 589 | NASDAQ:MRCY | MRCY | 未提供 | Q6818651 |
| 590 | NASDAQ:MRLN | MRLN | 未提供 | — |
| 591 | NASDAQ:MRTN | MRTN | 未提供 | — |
| 592 | NASDAQ:MRVL | MRVL | 未提供 | Q1347782 |
| 593 | NASDAQ:MSCC | MSCC | 未提供 | Q6840048 |
| 594 | NASDAQ:MSEX | MSEX | 未提供 | Q17113791 |
| 595 | NASDAQ:MSFG | MSFG | 未提供 | Q8178679 |
| 596 | NASDAQ:MSFT | MSFT | Microsoft | Q2283 |
| 597 | NASDAQ:MSTR | MSTR | 未提供 | Q1192211 |
| 598 | NASDAQ:MTEX | MTEX | 未提供 | Q3372572 |
| 599 | NASDAQ:MTGE | MTGE | 未提供 | — |
| 600 | NASDAQ:MTRX | MTRX | 未提供 | — |
| 601 | NASDAQ:MTSC | MTSC | 未提供 | Q1881941 |
| 602 | NASDAQ:MTSI | MTSI | 未提供 | — |
| 603 | NASDAQ:MU | MU | Micron Technology | Q1197548 |
| 604 | NASDAQ:MXIM | MXIM | 未提供 | Q1784671 |
| 605 | NASDAQ:MYGN | MYGN | 未提供 | Q3331496 |
| 606 | NASDAQ:MYRG | MYRG | 未提供 | — |
| 607 | NASDAQ:NANO | NANO | 未提供 | — |
| 608 | NASDAQ:NATH | NATH | 未提供 | Q1319505 |
| 609 | NASDAQ:NATI | NATI | 未提供 | Q1544413 |
| 610 | NASDAQ:NATR | NATR | 未提供 | Q4045079 |
| 611 | NASDAQ:NAVG | NAVG | 未提供 | — |
| 612 | NASDAQ:NBIX | NBIX | 未提供 | Q7002174 |
| 613 | NASDAQ:NBTB | NBTB | 未提供 | Q6952868 |
| 614 | NASDAQ:NCMI | NCMI | 未提供 | Q6971495 |
| 615 | NASDAQ:NDAQ | NDAQ | Nasdaq | Q1472929 |
| 616 | NASDAQ:NDSN | NDSN | 未提供 | Q1998662 |
| 617 | NASDAQ:NEOG | NEOG | 未提供 | Q17114199 |
| 618 | NASDAQ:NEWT | NEWT | 未提供 | — |
| 619 | NASDAQ:NFLX | NFLX | Netflix | Q907311 |
| 620 | NASDAQ:NICE | NICE | 未提供 | Q2091479 |
| 621 | NASDAQ:NICK | NICK | 未提供 | — |
| 622 | NASDAQ:NKSH | NKSH | 未提供 | — |
| 623 | NASDAQ:NKTR | NKTR | 未提供 | Q17114308 |
| 624 | NASDAQ:NLNK | NLNK | 未提供 | — |
| 625 | NASDAQ:NNBR | NNBR | 未提供 | — |
| 626 | NASDAQ:NOVT | NOVT | 未提供 | — |
| 627 | NASDAQ:NRIM | NRIM | 未提供 | — |
| 628 | NASDAQ:NSIT | NSIT | 未提供 | Q6038080 |
| 629 | NASDAQ:NTAP | NTAP | NetApp | Q727883 |
| 630 | NASDAQ:NTCT | NTCT | 未提供 | Q6998317 |
| 631 | NASDAQ:NTES | NTES | 未提供 | Q196259 |
| 632 | NASDAQ:NTGR | NTGR | 未提供 | Q979690 |
| 633 | NASDAQ:NTRI | NTRI | 未提供 | Q7070498 |
| 634 | NASDAQ:NTRS | NTRS | Northern Trust | Q970695 |
| 635 | NASDAQ:NUAN | NUAN | 未提供 | Q1137289 |
| 636 | NASDAQ:NUVA | NUVA | 未提供 | Q7067716 |
| 637 | NASDAQ:NVDA | NVDA | Nvidia | Q182477 |
| 638 | NASDAQ:NVEC | NVEC | 未提供 | — |
| 639 | NASDAQ:NVMI | NVMI | 未提供 | Q7064069 |
| 640 | NASDAQ:NWBI | NWBI | 未提供 | Q7060191 |
| 641 | NASDAQ:NWPX | NWPX | 未提供 | — |
| 642 | NASDAQ:NXPI | NXPI | NXP | Q1155668 |
| 643 | NASDAQ:NXST | NXST | 未提供 | Q7020906 |
| 644 | NASDAQ:NXTM | NXTM | 未提供 | Q7070744 |
| 645 | NASDAQ:NYNY | NYNY | 未提供 | Q5374115 |
| 646 | NASDAQ:OCFC | OCFC | 未提供 | — |
| 647 | NASDAQ:ODFL | ODFL | Old Dominion Freight Line | Q7083862 |
| 648 | NASDAQ:OFIX | OFIX | 未提供 | — |
| 649 | NASDAQ:OFLX | OFLX | 未提供 | — |
| 650 | NASDAQ:OFS | OFS | 未提供 | — |
| 651 | NASDAQ:OLBK | OLBK | 未提供 | — |
| 652 | NASDAQ:OLED | OLED | 未提供 | Q3116307 |
| 653 | NASDAQ:OMAB | OMAB | 未提供 | Q5612176 |
| 654 | NASDAQ:OMCL | OMCL | 未提供 | — |
| 655 | NASDAQ:ON | ON | 未提供 | Q1671509 |
| 656 | NASDAQ:ONB | ONB | 未提供 | Q7084595 |
| 657 | NASDAQ:ONEQ | ONEQ | 未提供 | — |
| 658 | NASDAQ:ORBK | ORBK | 未提供 | Q846743 |
| 659 | NASDAQ:ORIT | ORIT | 未提供 | — |
| 660 | NASDAQ:ORLY | ORLY | O'Reilly Automotive | Q7071951 |
| 661 | NASDAQ:ORRF | ORRF | 未提供 | — |
| 662 | NASDAQ:OSIS | OSIS | 未提供 | Q7572694 |
| 663 | NASDAQ:OSTK | OSTK | 未提供 | Q3301958 |
| 664 | NASDAQ:OTEX | OTEX | 未提供 | Q5515077 |
| 665 | NASDAQ:OTTR | OTTR | 未提供 | Q7109415 |
| 666 | NASDAQ:OXLC | OXLC | 未提供 | — |
| 667 | NASDAQ:OZRK | OZRK | 未提供 | — |
| 668 | NASDAQ:PAAS | PAAS | 未提供 | Q7129613 |
| 669 | NASDAQ:PACW | PACW | 未提供 | Q17143144 |
| 670 | NASDAQ:PAGG | PAGG | 未提供 | — |
| 671 | NASDAQ:PAYX | PAYX | Paychex | Q946080 |
| 672 | NASDAQ:PBCT | PBCT | People's United Financial | Q7165802 |
| 673 | NASDAQ:PCAR | PCAR | Paccar | Q1477661 |
| 674 | NASDAQ:PCH | PCH | 未提供 | Q7234972 |
| 675 | NASDAQ:PCLN | PCLN | 未提供 | Q18674747 |
| 676 | NASDAQ:PCMI | PCMI | 未提供 | Q7118707 |
| 677 | NASDAQ:PCO | PCO | 未提供 | Q3792482 |
| 678 | NASDAQ:PCOM | PCOM | 未提供 | — |
| 679 | NASDAQ:PCRX | PCRX | 未提供 | — |
| 680 | NASDAQ:PDCE | PDCE | 未提供 | — |
| 681 | NASDAQ:PDCO | PDCO | 未提供 | Q538203 |
| 682 | NASDAQ:PDFS | PDFS | 未提供 | Q7118809 |
| 683 | NASDAQ:PDP | PDP | 未提供 | — |
| 684 | NASDAQ:PEBO | PEBO | 未提供 | — |
| 685 | NASDAQ:PEGA | PEGA | 未提供 | Q7160600 |
| 686 | NASDAQ:PERY | PERY | 未提供 | Q7169774 |
| 687 | NASDAQ:PETS | PETS | 未提供 | Q7171526 |
| 688 | NASDAQ:PEY | PEY | 未提供 | — |
| 689 | NASDAQ:PEZ | PEZ | 未提供 | — |
| 690 | NASDAQ:PFBC | PFBC | 未提供 | Q7239836 |
| 691 | NASDAQ:PFBI | PFBI | 未提供 | — |
| 692 | NASDAQ:PFF | PFF | 未提供 | — |
| 693 | NASDAQ:PFI | PFI | 未提供 | — |
| 694 | NASDAQ:PFIN | PFIN | 未提供 | — |
| 695 | NASDAQ:PFLT | PFLT | 未提供 | — |
| 696 | NASDAQ:PFM | PFM | 未提供 | — |
| 697 | NASDAQ:PFPT | PFPT | 未提供 | Q4047452 |
| 698 | NASDAQ:PGC | PGC | 未提供 | — |
| 699 | NASDAQ:PGJ | PGJ | 未提供 | — |
| 700 | NASDAQ:PHIIK | PHIIK | 未提供 | — |
| 701 | NASDAQ:PHO | PHO | 未提供 | — |
| 702 | NASDAQ:PICO | PICO | 未提供 | — |
| 703 | NASDAQ:PID | PID | 未提供 | — |
| 704 | NASDAQ:PIE | PIE | 未提供 | — |
| 705 | NASDAQ:PIO | PIO | 未提供 | — |
| 706 | NASDAQ:PIZ | PIZ | 未提供 | — |
| 707 | NASDAQ:PKOH | PKOH | 未提供 | — |
| 708 | NASDAQ:PKW | PKW | 未提供 | — |
| 709 | NASDAQ:PLAB | PLAB | 未提供 | Q7187948 |
| 710 | NASDAQ:PLCE | PLCE | 未提供 | Q3520257 |
| 711 | NASDAQ:PLPC | PLPC | 未提供 | — |
| 712 | NASDAQ:PLUS | PLUS | 未提供 | Q16992684 |
| 713 | NASDAQ:PLW | PLW | 未提供 | — |
| 714 | NASDAQ:PLXS | PLXS | 未提供 | Q2099437 |
| 715 | NASDAQ:PMBC | PMBC | 未提供 | — |
| 716 | NASDAQ:PMD | PMD | 未提供 | Q7256219 |
| 717 | NASDAQ:PNFP | PNFP | 未提供 | Q7196294 |
| 718 | NASDAQ:PNQI | PNQI | 未提供 | — |
| 719 | NASDAQ:PODD | PODD | 未提供 | — |
| 720 | NASDAQ:POOL | POOL | Pool Corporation | — |
| 721 | NASDAQ:POWI | POWI | 未提供 | — |
| 722 | NASDAQ:POWL | POWL | 未提供 | Q7236078 |
| 723 | NASDAQ:PPBI | PPBI | 未提供 | — |
| 724 | NASDAQ:PPC | PPC | 未提供 | Q3388421 |
| 725 | NASDAQ:PPH | PPH | 未提供 | — |
| 726 | NASDAQ:PRAA | PRAA | 未提供 | Q19868026 |
| 727 | NASDAQ:PRFT | PRFT | 未提供 | Q7168146 |
| 728 | NASDAQ:PRFZ | PRFZ | 未提供 | — |
| 729 | NASDAQ:PRGS | PRGS | 未提供 | — |
| 730 | NASDAQ:PRIM | PRIM | 未提供 | Q20714950 |
| 731 | NASDAQ:PRN | PRN | 未提供 | — |
| 732 | NASDAQ:PROV | PROV | 未提供 | — |
| 733 | NASDAQ:PRSC | PRSC | 未提供 | — |
| 734 | NASDAQ:PRTA | PRTA | 未提供 | — |
| 735 | NASDAQ:PRTK | PRTK | 未提供 | — |
| 736 | NASDAQ:PSAU | PSAU | 未提供 | — |
| 737 | NASDAQ:PSCC | PSCC | 未提供 | — |
| 738 | NASDAQ:PSCD | PSCD | 未提供 | — |
| 739 | NASDAQ:PSCE | PSCE | 未提供 | — |
| 740 | NASDAQ:PSCF | PSCF | 未提供 | — |
| 741 | NASDAQ:PSCH | PSCH | 未提供 | — |
| 742 | NASDAQ:PSCI | PSCI | 未提供 | — |
| 743 | NASDAQ:PSCT | PSCT | 未提供 | — |
| 744 | NASDAQ:PSCU | PSCU | 未提供 | — |
| 745 | NASDAQ:PSEC | PSEC | 未提供 | — |
| 746 | NASDAQ:PSL | PSL | 未提供 | — |
| 747 | NASDAQ:PSMT | PSMT | 未提供 | Q7242563 |
| 748 | NASDAQ:PSTB | PSTB | 未提供 | Q7138046 |
| 749 | NASDAQ:PTC | PTC | PTC | — |
| 750 | NASDAQ:PTEN | PTEN | 未提供 | Q3370152 |
| 751 | NASDAQ:PTF | PTF | 未提供 | — |
| 752 | NASDAQ:PTH | PTH | 未提供 | — |
| 753 | NASDAQ:PTSI | PTSI | 未提供 | Q7118151 |
| 754 | NASDAQ:PUI | PUI | 未提供 | — |
| 755 | NASDAQ:PWOD | PWOD | 未提供 | — |
| 756 | NASDAQ:PXI | PXI | 未提供 | — |
| 757 | NASDAQ:PYZ | PYZ | 未提供 | — |
| 758 | NASDAQ:PZZA | PZZA | 未提供 | Q2759586 |
| 759 | NASDAQ:QABA | QABA | 未提供 | — |
| 760 | NASDAQ:QADA | QADA | 未提供 | Q2899735 |
| 761 | NASDAQ:QCLN | QCLN | 未提供 | — |
| 762 | NASDAQ:QCOM | QCOM | Qualcomm | Q544847 |
| 763 | NASDAQ:QCRH | QCRH | 未提供 | — |
| 764 | NASDAQ:QDEL | QDEL | 未提供 | Q15917582 |
| 765 | NASDAQ:QGEN | QGEN | 未提供 | Q315287 |
| 766 | NASDAQ:QLYS | QLYS | 未提供 | Q7268802 |
| 767 | NASDAQ:QQEW | QQEW | 未提供 | — |
| 768 | NASDAQ:QQQ | QQQ | 未提供 | Q507306 |
| 769 | NASDAQ:QQXT | QQXT | 未提供 | — |
| 770 | NASDAQ:QSII | QSII | 未提供 | Q17109731 |
| 771 | NASDAQ:QTEC | QTEC | 未提供 | — |
| 772 | NASDAQ:QVCA | QVCA | 未提供 | Q6541837 |
| 773 | NASDAQ:RAIL | RAIL | 未提供 | Q5501354 |
| 774 | NASDAQ:RAVN | RAVN | 未提供 | Q7296403 |
| 775 | NASDAQ:RBCAA | RBCAA | 未提供 | Q7314387 |
| 776 | NASDAQ:RCII | RCII | 未提供 | Q7313497 |
| 777 | NASDAQ:RCKY | RCKY | 未提供 | Q7355744 |
| 778 | NASDAQ:RDI | RDI | 未提供 | — |
| 779 | NASDAQ:RDWR | RDWR | 未提供 | Q2907743 |
| 780 | NASDAQ:RECN | RECN | 未提供 | Q2228136 |
| 781 | NASDAQ:REGI | REGI | 未提供 | — |
| 782 | NASDAQ:REGN | REGN | Regeneron Pharmaceuticals | Q7308054 |
| 783 | NASDAQ:REIS | REIS | 未提供 | — |
| 784 | NASDAQ:RGEN | RGEN | 未提供 | Q17078090 |
| 785 | NASDAQ:RGLD | RGLD | 未提供 | Q7374224 |
| 786 | NASDAQ:RING | RING | 未提供 | — |
| 787 | NASDAQ:RNET | RNET | 未提供 | — |
| 788 | NASDAQ:RNST | RNST | 未提供 | — |
| 789 | NASDAQ:ROCK | ROCK | 未提供 | Q20656950 |
| 790 | NASDAQ:ROIC | ROIC | 未提供 | — |
| 791 | NASDAQ:ROLL | ROLL | 未提供 | — |
| 792 | NASDAQ:ROST | ROST | Ross Stores | Q3442791 |
| 793 | NASDAQ:RP | RP | 未提供 | — |
| 794 | NASDAQ:RPXC | RPXC | 未提供 | Q7277601 |
| 795 | NASDAQ:RRGB | RRGB | 未提供 | Q7304886 |
| 796 | NASDAQ:RUSHA | RUSHA | 未提供 | — |
| 797 | NASDAQ:RUSHB | RUSHB | 未提供 | — |
| 798 | NASDAQ:RUTH | RUTH | 未提供 | Q7382830 |
| 799 | NASDAQ:RYAAY | RYAAY | 未提供 | Q170614 |
| 800 | NASDAQ:SAFM | SAFM | 未提供 | Q20022777 |
| 801 | NASDAQ:SAFT | SAFT | 未提供 | — |
| 802 | NASDAQ:SAIA | SAIA | 未提供 | Q7399844 |
| 803 | NASDAQ:SANM | SANM | 未提供 | Q426583 |
| 804 | NASDAQ:SASR | SASR | 未提供 | Q17109642 |
| 805 | NASDAQ:SATS | SATS | 未提供 | Q1280748 |
| 806 | NASDAQ:SAVE | SAVE | 未提供 | Q1190598 |
| 807 | NASDAQ:SBCF | SBCF | 未提供 | — |
| 808 | NASDAQ:SBGI | SBGI | 未提供 | Q2288792 |
| 809 | NASDAQ:SBNY | SBNY | 未提供 | — |
| 810 | NASDAQ:SBRA | SBRA | 未提供 | — |
| 811 | NASDAQ:SBSI | SBSI | 未提供 | — |
| 812 | NASDAQ:SBUX | SBUX | Starbucks | Q37158 |
| 813 | NASDAQ:SCHL | SCHL | 未提供 | Q1155729 |
| 814 | NASDAQ:SCHN | SCHN | 未提供 | Q7431916 |
| 815 | NASDAQ:SCSC | SCSC | 未提供 | — |
| 816 | NASDAQ:SCVL | SCVL | 未提供 | Q7500007 |
| 817 | NASDAQ:SCZ | SCZ | 未提供 | — |
| 818 | NASDAQ:SEIC | SEIC | 未提供 | Q7389686 |
| 819 | NASDAQ:SENEA | SENEA | 未提供 | — |
| 820 | NASDAQ:SFLY | SFLY | 未提供 | Q7505672 |
| 821 | NASDAQ:SFNC | SFNC | 未提供 | — |
| 822 | NASDAQ:SGC | SGC | 未提供 | — |
| 823 | NASDAQ:SGEN | SGEN | 未提供 | Q17155112 |
| 824 | NASDAQ:SHBI | SHBI | 未提供 | — |
| 825 | NASDAQ:SHEN | SHEN | 未提供 | — |
| 826 | NASDAQ:SHLM | SHLM | 未提供 | — |
| 827 | NASDAQ:SHOO | SHOO | 未提供 | — |
| 828 | NASDAQ:SHPG | SHPG | 未提供 | Q905049 |
| 829 | NASDAQ:SHV | SHV | 未提供 | — |
| 830 | NASDAQ:SHY | SHY | 未提供 | — |
| 831 | NASDAQ:SIFI | SIFI | 未提供 | — |
| 832 | NASDAQ:SIGI | SIGI | 未提供 | Q7447744 |
| 833 | NASDAQ:SILC | SILC | 未提供 | — |
| 834 | NASDAQ:SIMO | SIMO | 未提供 | — |
| 835 | NASDAQ:SINA | SINA | 未提供 | Q938668 |
| 836 | NASDAQ:SIR | SIR | 未提供 | — |
| 837 | NASDAQ:SIVB | SIVB | SVB Financial | Q17112054 |
| 838 | NASDAQ:SKYW | SKYW | 未提供 | Q7537214 |
| 839 | NASDAQ:SKYY | SKYY | 未提供 | — |
| 840 | NASDAQ:SLAB | SLAB | 未提供 | Q2286044 |
| 841 | NASDAQ:SLGN | SLGN | 未提供 | — |
| 842 | NASDAQ:SLRC | SLRC | 未提供 | — |
| 843 | NASDAQ:SMCI | SMCI | 未提供 | Q1817174 |
| 844 | NASDAQ:SMTC | SMTC | 未提供 | Q3459777 |
| 845 | NASDAQ:SNBC | SNBC | 未提供 | Q7638414 |
| 846 | NASDAQ:SNBR | SNBR | 未提供 | — |
| 847 | NASDAQ:SNCR | SNCR | 未提供 | — |
| 848 | NASDAQ:SNH | SNH | 未提供 | — |
| 849 | NASDAQ:SNHY | SNHY | 未提供 | — |
| 850 | NASDAQ:SNPS | SNPS | Synopsys | Q2303478 |
| 851 | NASDAQ:SOCL | SOCL | 未提供 | — |
| 852 | NASDAQ:SODA | SODA | 未提供 | Q683006 |
| 853 | NASDAQ:SOHU | SOHU | 未提供 | Q965133 |
| 854 | NASDAQ:SONA | SONA | 未提供 | — |
| 855 | NASDAQ:SONC | SONC | 未提供 | Q7561808 |
| 856 | NASDAQ:SOXX | SOXX | 未提供 | Q2443320 |
| 857 | NASDAQ:SP | SP | 未提供 | Q7598289 |
| 858 | NASDAQ:SPIL | SPIL | 未提供 | — |
| 859 | NASDAQ:SPLK | SPLK | 未提供 | — |
| 860 | NASDAQ:SPSC | SPSC | 未提供 | — |
| 861 | NASDAQ:SPTN | SPTN | 未提供 | Q7573912 |
| 862 | NASDAQ:SPWR | SPWR | 未提供 | Q645525 |
| 863 | NASDAQ:SQQQ | SQQQ | 未提供 | — |
| 864 | NASDAQ:SRCE | SRCE | 未提供 | — |
| 865 | NASDAQ:SRCL | SRCL | 未提供 | Q3973262 |
| 866 | NASDAQ:SRDX | SRDX | 未提供 | — |
| 867 | NASDAQ:SSNC | SSNC | 未提供 | Q7392916 |
| 868 | NASDAQ:SSYS | SSYS | 未提供 | Q14705393 |
| 869 | NASDAQ:STAA | STAA | 未提供 | — |
| 870 | NASDAQ:STBA | STBA | 未提供 | — |
| 871 | NASDAQ:STBZ | STBZ | 未提供 | — |
| 872 | NASDAQ:STFC | STFC | 未提供 | — |
| 873 | NASDAQ:STLD | STLD | 未提供 | Q7605667 |
| 874 | NASDAQ:STMP | STMP | 未提供 | Q7597506 |
| 875 | NASDAQ:STRA | STRA | 未提供 | — |
| 876 | NASDAQ:STRT | STRT | 未提供 | — |
| 877 | NASDAQ:STX | STX | Seagate Technology | Q705392 |
| 878 | NASDAQ:SUNS | SUNS | 未提供 | — |
| 879 | NASDAQ:SVVC | SVVC | 未提供 | — |
| 880 | NASDAQ:SWIR | SWIR | 未提供 | Q13529476 |
| 881 | NASDAQ:SWKS | SWKS | Skyworks Solutions | Q7538146 |
| 882 | NASDAQ:SYBT | SYBT | 未提供 | — |
| 883 | NASDAQ:SYKE | SYKE | 未提供 | Q2373884 |
| 884 | NASDAQ:SYMC | SYMC | 未提供 | Q7131487 |
| 885 | NASDAQ:SYNA | SYNA | 未提供 | Q1571490 |
| 886 | NASDAQ:SYNL | SYNL | 未提供 | — |
| 887 | NASDAQ:SYNT | SYNT | 未提供 | Q7662679 |
| 888 | NASDAQ:TACT | TACT | 未提供 | — |
| 889 | NASDAQ:TAX | TAX | 未提供 | Q6541978 |
| 890 | NASDAQ:TBNK | TBNK | 未提供 | — |
| 891 | NASDAQ:TCBI | TCBI | 未提供 | — |
| 892 | NASDAQ:TCBK | TCBK | 未提供 | — |
| 893 | NASDAQ:TCPC | TCPC | 未提供 | — |
| 894 | NASDAQ:TCRD | TCRD | 未提供 | — |
| 895 | NASDAQ:TCX | TCX | 未提供 | Q2087612 |
| 896 | NASDAQ:TDIV | TDIV | 未提供 | — |
| 897 | NASDAQ:TECD | TECD | 未提供 | Q1715931 |
| 898 | NASDAQ:TECH | TECH | Bio-Techne | Q19873469 |
| 899 | NASDAQ:TESS | TESS | 未提供 | — |
| 900 | NASDAQ:TFSL | TFSL | 未提供 | — |
| 901 | NASDAQ:THFF | THFF | 未提供 | Q5453008 |
| 902 | NASDAQ:THRM | THRM | 未提供 | — |
| 903 | NASDAQ:TITN | TITN | 未提供 | Q7809764 |
| 904 | NASDAQ:TIVO | TIVO | 未提供 | Q7799971 |
| 905 | NASDAQ:TOWN | TOWN | 未提供 | — |
| 906 | NASDAQ:TQQQ | TQQQ | 未提供 | — |
| 907 | NASDAQ:TREE | TREE | 未提供 | Q6522339 |
| 908 | NASDAQ:TRIP | TRIP | 未提供 | Q1770710 |
| 909 | NASDAQ:TRMB | TRMB | Trimble | Q17079979 |
| 910 | NASDAQ:TRMK | TRMK | 未提供 | — |
| 911 | NASDAQ:TROW | TROW | T. Rowe Price | Q3511946 |
| 912 | NASDAQ:TRS | TRS | 未提供 | — |
| 913 | NASDAQ:TRST | TRST | 未提供 | Q7848087 |
| 914 | NASDAQ:TSCO | TSCO | Tractor Supply Company | Q15109925 |
| 915 | NASDAQ:TSLA | TSLA | Tesla | Q7555824 |
| 916 | NASDAQ:TTEC | TTEC | 未提供 | Q7695737 |
| 917 | NASDAQ:TTEK | TTEK | 未提供 | Q7706196 |
| 918 | NASDAQ:TTMI | TTMI | 未提供 | — |
| 919 | NASDAQ:TTS | TTS | 未提供 | — |
| 920 | NASDAQ:TTWO | TTWO | Take-Two Interactive | Q94933 |
| 921 | NASDAQ:TUR | TUR | 未提供 | — |
| 922 | NASDAQ:TVTY | TVTY | 未提供 | — |
| 923 | NASDAQ:TWIN | TWIN | 未提供 | Q7858146 |
| 924 | NASDAQ:TXN | TXN | Texas Instruments | Q193412 |
| 925 | NASDAQ:TXRH | TXRH | 未提供 | Q7707945 |
| 926 | NASDAQ:TYPE | TYPE | 未提供 | Q1583606 |
| 927 | NASDAQ:UBNK | UBNK | 未提供 | Q16959074 |
| 928 | NASDAQ:UBNT | UBNT | 未提供 | Q7876533 |
| 929 | NASDAQ:UBSH | UBSH | 未提供 | — |
| 930 | NASDAQ:UBSI | UBSI | 未提供 | Q16920636 |
| 931 | NASDAQ:UCBI | UCBI | 未提供 | Q7887635 |
| 932 | NASDAQ:UEIC | UEIC | 未提供 | — |
| 933 | NASDAQ:UEPS | UEPS | 未提供 | — |
| 934 | NASDAQ:UFCS | UFCS | 未提供 | — |
| 935 | NASDAQ:UFPI | UFPI | 未提供 | — |
| 936 | NASDAQ:UFPT | UFPT | 未提供 | Q17145997 |
| 937 | NASDAQ:UG | UG | 未提供 | — |
| 938 | NASDAQ:UGLD | UGLD | 未提供 | — |
| 939 | NASDAQ:UHAL | UHAL | 未提供 | — |
| 940 | NASDAQ:UIHC | UIHC | 未提供 | — |
| 941 | NASDAQ:ULH | ULH | 未提供 | — |
| 942 | NASDAQ:ULTA | ULTA | Ulta Beauty | Q7880076 |
| 943 | NASDAQ:ULTI | ULTI | 未提供 | Q7880250 |
| 944 | NASDAQ:UMBF | UMBF | 未提供 | Q7865088 |
| 945 | NASDAQ:UMPQ | UMPQ | 未提供 | Q7881772 |
| 946 | NASDAQ:UNFI | UNFI | 未提供 | Q7888979 |
| 947 | NASDAQ:URBN | URBN | 未提供 | Q3552193 |
| 948 | NASDAQ:USAP | USAP | 未提供 | — |
| 949 | NASDAQ:USCR | USCR | 未提供 | — |
| 950 | NASDAQ:USLM | USLM | 未提供 | — |
| 951 | NASDAQ:UTHR | UTHR | 未提供 | — |
| 952 | NASDAQ:UTMD | UTMD | 未提供 | — |
| 953 | NASDAQ:UVSP | UVSP | 未提供 | — |
| 954 | NASDAQ:VCIT | VCIT | 未提供 | — |
| 955 | NASDAQ:VCLT | VCLT | 未提供 | — |
| 956 | NASDAQ:VCSH | VCSH | 未提供 | — |
| 957 | NASDAQ:VDSI | VDSI | 未提供 | — |
| 958 | NASDAQ:VECO | VECO | 未提供 | Q1665653 |
| 959 | NASDAQ:VGIT | VGIT | 未提供 | — |
| 960 | NASDAQ:VGLT | VGLT | 未提供 | — |
| 961 | NASDAQ:VGSH | VGSH | 未提供 | — |
| 962 | NASDAQ:VIA | VIA | 未提供 | Q214346 |
| 963 | NASDAQ:VIAB | VIAB | 未提供 | — |
| 964 | NASDAQ:VIVO | VIVO | 未提供 | Q17113726 |
| 965 | NASDAQ:VLGEA | VLGEA | 未提供 | Q7501097 |
| 966 | NASDAQ:VMBS | VMBS | 未提供 | — |
| 967 | NASDAQ:VNQI | VNQI | 未提供 | — |
| 968 | NASDAQ:VOD | VOD | 未提供 | Q122141 |
| 969 | NASDAQ:VONE | VONE | 未提供 | — |
| 970 | NASDAQ:VONG | VONG | 未提供 | — |
| 971 | NASDAQ:VONV | VONV | 未提供 | — |
| 972 | NASDAQ:VRA | VRA | 未提供 | Q7920749 |
| 973 | NASDAQ:VRNT | VRNT | 未提供 | Q7921363 |
| 974 | NASDAQ:VRSK | VRSK | Verisk Analytics | Q7921370 |
| 975 | NASDAQ:VRSN | VRSN | Verisign | Q734338 |
| 976 | NASDAQ:VRTS | VRTS | 未提供 | Q7756963 |
| 977 | NASDAQ:VRTU | VRTU | 未提供 | Q2516971 |
| 978 | NASDAQ:VRTX | VRTX | Vertex Pharmaceuticals | Q2744099 |
| 979 | NASDAQ:VSAT | VSAT | 未提供 | Q7924358 |
| 980 | NASDAQ:VSEC | VSEC | 未提供 | Q7907548 |
| 981 | NASDAQ:VTHR | VTHR | 未提供 | — |
| 982 | NASDAQ:VTIP | VTIP | 未提供 | — |
| 983 | NASDAQ:VTWG | VTWG | 未提供 | — |
| 984 | NASDAQ:VTWO | VTWO | 未提供 | — |
| 985 | NASDAQ:VTWV | VTWV | 未提供 | — |
| 986 | NASDAQ:VXUS | VXUS | 未提供 | — |
| 987 | NASDAQ:WABC | WABC | 未提供 | Q7987101 |
| 988 | NASDAQ:WAFD | WAFD | 未提供 | Q7971859 |
| 989 | NASDAQ:WASH | WASH | 未提供 | — |
| 990 | NASDAQ:WDAY | WDAY | 未提供 | — |
| 991 | NASDAQ:WDC | WDC | Western Digital | Q738770 |
| 992 | NASDAQ:WDFC | WDFC | 未提供 | — |
| 993 | NASDAQ:WEB | WEB | 未提供 | — |
| 994 | NASDAQ:WERN | WERN | 未提供 | Q7983382 |
| 995 | NASDAQ:WETF | WETF | 未提供 | Q8027406 |
| 996 | NASDAQ:WEYS | WEYS | 未提供 | — |
| 997 | NASDAQ:WHF | WHF | 未提供 | — |
| 998 | NASDAQ:WIFI | WIFI | 未提供 | Q4938392 |
| 999 | NASDAQ:WINA | WINA | 未提供 | — |
| 1000 | NASDAQ:WIRE | WIRE | 未提供 | — |
| 1001 | NASDAQ:WLFC | WLFC | 未提供 | — |
| 1002 | NASDAQ:WLTW | WLTW | Willis Towers Watson | — |
| 1003 | NASDAQ:WNEB | WNEB | 未提供 | — |
| 1004 | NASDAQ:WOOD | WOOD | 未提供 | — |
| 1005 | NASDAQ:WRLD | WRLD | 未提供 | — |
| 1006 | NASDAQ:WSBC | WSBC | 未提供 | Q7983629 |
| 1007 | NASDAQ:WSBF | WSBF | 未提供 | — |
| 1008 | NASDAQ:WSTG | WSTG | 未提供 | Q6545346 |
| 1009 | NASDAQ:WTBA | WTBA | 未提供 | — |
| 1010 | NASDAQ:WTFC | WTFC | 未提供 | — |
| 1011 | NASDAQ:WWD | WWD | 未提供 | Q8033550 |
| 1012 | NASDAQ:WYNN | WYNN | Wynn Resorts | Q2274089 |
| 1013 | NASDAQ:XBKS | XBKS | 未提供 | — |
| 1014 | NASDAQ:XIV | XIV | 未提供 | — |
| 1015 | NASDAQ:XLNX | XLNX | Xilinx | Q1046482 |
| 1016 | NASDAQ:XRAY | XRAY | Dentsply Sirona | Q834888 |
| 1017 | NASDAQ:YNDX | YNDX | 未提供 | Q5281 |
| 1018 | NASDAQ:YORW | YORW | 未提供 | — |
| 1019 | NASDAQ:YRCW | YRCW | 未提供 | Q8046013 |
| 1020 | NASDAQ:YY | YY | 未提供 | Q8046104 |
| 1021 | NASDAQ:ZBRA | ZBRA | Zebra Technologies | Q2443609 |
| 1022 | NASDAQ:ZEUS | ZEUS | 未提供 | Q7089136 |
| 1023 | NASDAQ:ZION | ZION | Zions Bancorp | Q8072579 |
| 1024 | NASDAQ:ZIV | ZIV | 未提供 | — |
| 1025 | NASDAQ:ZUMZ | ZUMZ | 未提供 | Q8075252 |

### SP500：474 只

| array_index | 建议 stock_uid | 股票代码 ticker | 公司名称（本地快照） | 源 Wikidata QID |
|---:|---|---|---|---|
| 0 | SP500:POOL | POOL | Pool Corporation | — |
| 1 | SP500:MKC | MKC | McCormick & Company | — |
| 2 | SP500:AOS | AOS | A. O. Smith | — |
| 3 | SP500:AXP | AXP | American Express | — |
| 4 | SP500:CME | CME | CME Group | — |
| 5 | SP500:EQR | EQR | Equity Residential | — |
| 6 | SP500:BKR | BKR | Baker Hughes | — |
| 7 | SP500:FDX | FDX | FedEx | — |
| 8 | SP500:NCLH | NCLH | Norwegian Cruise Line Holdings | — |
| 9 | SP500:MET | MET | MetLife | — |
| 10 | SP500:MRK | MRK | Merck & Co. | — |
| 11 | SP500:MAA | MAA | Mid-America Apartments | — |
| 12 | SP500:ISRG | ISRG | Intuitive Surgical | — |
| 13 | SP500:GOOGL | GOOGL | Alphabet (Class A) | — |
| 14 | SP500:PEAK | PEAK | Healthpeak Properties | — |
| 15 | SP500:CMCSA | CMCSA | Comcast | — |
| 16 | SP500:TT | TT | Trane Technologies | — |
| 17 | SP500:JPM | JPM | JPMorgan Chase | — |
| 18 | SP500:REGN | REGN | Regeneron Pharmaceuticals | — |
| 19 | SP500:DHI | DHI | D. R. Horton | — |
| 20 | SP500:HUM | HUM | Humana | — |
| 21 | SP500:MDT | MDT | Medtronic | — |
| 22 | SP500:DIS | DIS | The Walt Disney Company | — |
| 23 | SP500:DE | DE | Deere & Co. | — |
| 24 | SP500:F | F | Ford | — |
| 25 | SP500:C | C | Citigroup | — |
| 26 | SP500:EXC | EXC | Exelon | — |
| 27 | SP500:SCHW | SCHW | Charles Schwab Corporation | — |
| 28 | SP500:WYNN | WYNN | Wynn Resorts | — |
| 29 | SP500:PLD | PLD | Prologis | — |
| 30 | SP500:AAL | AAL | American Airlines Group | — |
| 31 | SP500:DVN | DVN | Devon Energy | — |
| 32 | SP500:AEE | AEE | Ameren Corp | — |
| 33 | SP500:IQV | IQV | IQVIA | — |
| 34 | SP500:AMZN | AMZN | Amazon | — |
| 35 | SP500:TDY | TDY | Teledyne Technologies | — |
| 36 | SP500:PENN | PENN | Penn National Gaming | — |
| 37 | SP500:IP | IP | International Paper | — |
| 38 | SP500:OMC | OMC | Omnicom Group | — |
| 39 | SP500:VFC | VFC | VF Corporation | — |
| 40 | SP500:MSCI | MSCI | MSCI | — |
| 41 | SP500:A | A | Agilent Technologies | — |
| 42 | SP500:USB | USB | U.S. Bancorp | — |
| 43 | SP500:UNH | UNH | UnitedHealth Group | — |
| 44 | SP500:IT | IT | Gartner | — |
| 45 | SP500:ETR | ETR | Entergy | — |
| 46 | SP500:DVA | DVA | DaVita | — |
| 47 | SP500:HSY | HSY | The Hershey Company | — |
| 48 | SP500:NEE | NEE | NextEra Energy | — |
| 49 | SP500:FFIV | FFIV | F5 Networks | — |
| 50 | SP500:BIIB | BIIB | Biogen | — |
| 51 | SP500:KO | KO | Coca-Cola Company | — |
| 52 | SP500:MSI | MSI | Motorola Solutions | — |
| 53 | SP500:KMB | KMB | Kimberly-Clark | — |
| 54 | SP500:FTNT | FTNT | Fortinet | — |
| 55 | SP500:EL | EL | Estée Lauder Companies | — |
| 56 | SP500:ABC | ABC | AmerisourceBergen | — |
| 57 | SP500:CF | CF | CF Industries | — |
| 58 | SP500:LKQ | LKQ | LKQ Corporation | — |
| 59 | SP500:TMUS | TMUS | T-Mobile US | — |
| 60 | SP500:ABT | ABT | Abbott Laboratories | — |
| 61 | SP500:PSX | PSX | Phillips 66 | — |
| 62 | SP500:LMT | LMT | Lockheed Martin | — |
| 63 | SP500:ESS | ESS | Essex Property Trust | — |
| 64 | SP500:APD | APD | Air Products & Chemicals | — |
| 65 | SP500:PSA | PSA | Public Storage | — |
| 66 | SP500:AMAT | AMAT | Applied Materials | — |
| 67 | SP500:HCA | HCA | HCA Healthcare | — |
| 68 | SP500:ORLY | ORLY | O'Reilly Automotive | — |
| 69 | SP500:KEY | KEY | KeyCorp | — |
| 70 | SP500:KLAC | KLAC | KLA Corporation | — |
| 71 | SP500:GL | GL | Globe Life | — |
| 72 | SP500:PCAR | PCAR | Paccar | — |
| 73 | SP500:CBOE | CBOE | Cboe Global Markets | — |
| 74 | SP500:PRU | PRU | Prudential Financial | — |
| 75 | SP500:FE | FE | FirstEnergy | — |
| 76 | SP500:LEG | LEG | Leggett & Platt | — |
| 77 | SP500:PHM | PHM | PulteGroup | — |
| 78 | SP500:NOC | NOC | Northrop Grumman | — |
| 79 | SP500:BBWI | BBWI | Bath & Body Works Inc. | — |
| 80 | SP500:MCK | MCK | McKesson Corporation | — |
| 81 | SP500:CPB | CPB | Campbell Soup | — |
| 82 | SP500:KR | KR | Kroger | — |
| 83 | SP500:DGX | DGX | Quest Diagnostics | — |
| 84 | SP500:TJX | TJX | TJX Companies | — |
| 85 | SP500:GRMN | GRMN | Garmin | — |
| 86 | SP500:PKG | PKG | Packaging Corporation of America | — |
| 87 | SP500:FISV | FISV | Fiserv | — |
| 88 | SP500:OXY | OXY | Occidental Petroleum | — |
| 89 | SP500:FITB | FITB | Fifth Third Bancorp | — |
| 90 | SP500:CBRE | CBRE | CBRE | — |
| 91 | SP500:WAB | WAB | Wabtec | — |
| 92 | SP500:KMX | KMX | CarMax | — |
| 93 | SP500:RSG | RSG | Republic Services | — |
| 94 | SP500:ADM | ADM | ADM | — |
| 95 | SP500:PEP | PEP | PepsiCo | — |
| 96 | SP500:IRM | IRM | Iron Mountain | — |
| 97 | SP500:LOW | LOW | Lowe's | — |
| 98 | SP500:CL | CL | Colgate-Palmolive | — |
| 99 | SP500:TXT | TXT | Textron | — |
| 100 | SP500:MCO | MCO | Moody's Corporation | — |
| 101 | SP500:DUK | DUK | Duke Energy | — |
| 102 | SP500:NKE | NKE | Nike | — |
| 103 | SP500:PAYX | PAYX | Paychex | — |
| 104 | SP500:IFF | IFF | International Flavors & Fragrances | — |
| 105 | SP500:MSFT | MSFT | Microsoft | — |
| 106 | SP500:CTRA | CTRA | Coterra | — |
| 107 | SP500:MA | MA | Mastercard | — |
| 108 | SP500:NWS | NWS | News Corp (Class B) | — |
| 109 | SP500:PVH | PVH | PVH | — |
| 110 | SP500:TSN | TSN | Tyson Foods | — |
| 111 | SP500:DXC | DXC | DXC Technology | — |
| 112 | SP500:VTRS | VTRS | Viatris | — |
| 113 | SP500:TWTR | TWTR | Twitter | — |
| 114 | SP500:MTD | MTD | Mettler Toledo | — |
| 115 | SP500:MLM | MLM | Martin Marietta Materials | — |
| 116 | SP500:SO | SO | Southern Company | — |
| 117 | SP500:ETN | ETN | Eaton Corporation | — |
| 118 | SP500:STT | STT | State Street Corporation | — |
| 119 | SP500:GD | GD | General Dynamics | — |
| 120 | SP500:CMS | CMS | CMS Energy | — |
| 121 | SP500:SRE | SRE | Sempra Energy | — |
| 122 | SP500:MMC | MMC | Marsh & McLennan | — |
| 123 | SP500:AIZ | AIZ | Assurant | — |
| 124 | SP500:NOW | NOW | ServiceNow | — |
| 125 | SP500:TGT | TGT | Target Corporation | — |
| 126 | SP500:BRK-B | BRK-B | Berkshire Hathaway | — |
| 127 | SP500:MRO | MRO | Marathon Oil | — |
| 128 | SP500:NXPI | NXPI | NXP | — |
| 129 | SP500:WFC | WFC | Wells Fargo | — |
| 130 | SP500:EXR | EXR | Extra Space Storage | — |
| 131 | SP500:NTAP | NTAP | NetApp | — |
| 132 | SP500:BDX | BDX | Becton Dickinson | — |
| 133 | SP500:AJG | AJG | Arthur J. Gallagher & Co. | — |
| 134 | SP500:WY | WY | Weyerhaeuser | — |
| 135 | SP500:EOG | EOG | EOG Resources | — |
| 136 | SP500:XYL | XYL | Xylem | — |
| 137 | SP500:PH | PH | Parker-Hannifin | — |
| 138 | SP500:IPG | IPG | Interpublic Group | — |
| 139 | SP500:DLTR | DLTR | Dollar Tree | — |
| 140 | SP500:TDG | TDG | TransDigm Group | — |
| 141 | SP500:LUV | LUV | Southwest Airlines | — |
| 142 | SP500:AFL | AFL | Aflac | — |
| 143 | SP500:AVGO | AVGO | Broadcom | — |
| 144 | SP500:LLY | LLY | Eli Lilly & Co | — |
| 145 | SP500:CAG | CAG | Conagra Brands | — |
| 146 | SP500:ODFL | ODFL | Old Dominion Freight Line | — |
| 147 | SP500:ZBH | ZBH | Zimmer Biomet | — |
| 148 | SP500:KMI | KMI | Kinder Morgan | — |
| 149 | SP500:LUMN | LUMN | Lumen Technologies | — |
| 150 | SP500:PGR | PGR | Progressive Corporation | — |
| 151 | SP500:ROL | ROL | Rollins | — |
| 152 | SP500:PNC | PNC | PNC Financial Services | — |
| 153 | SP500:DRE | DRE | Duke Realty Corp | — |
| 154 | SP500:EQIX | EQIX | Equinix | — |
| 155 | SP500:LRCX | LRCX | Lam Research | — |
| 156 | SP500:VMC | VMC | Vulcan Materials | — |
| 157 | SP500:TRMB | TRMB | Trimble | — |
| 158 | SP500:HPQ | HPQ | HP | — |
| 159 | SP500:ADP | ADP | Automatic Data Processing | — |
| 160 | SP500:ARE | ARE | Alexandria Real Estate Equities | — |
| 161 | SP500:PTC | PTC | PTC | — |
| 162 | SP500:HD | HD | Home Depot | — |
| 163 | SP500:NSC | NSC | Norfolk Southern | — |
| 164 | SP500:TXN | TXN | Texas Instruments | — |
| 165 | SP500:SEE | SEE | Sealed Air | — |
| 166 | SP500:WRB | WRB | W. R. Berkley Corporation | — |
| 167 | SP500:SBUX | SBUX | Starbucks | — |
| 168 | SP500:ADBE | ADBE | Adobe | — |
| 169 | SP500:DAL | DAL | Delta Air Lines | — |
| 170 | SP500:GPS | GPS | Gap | — |
| 171 | SP500:AMGN | AMGN | Amgen | — |
| 172 | SP500:JBHT | JBHT | J. B. Hunt | — |
| 173 | SP500:COF | COF | Capital One Financial | — |
| 174 | SP500:NRG | NRG | NRG Energy | — |
| 175 | SP500:CINF | CINF | Cincinnati Financial | — |
| 176 | SP500:WDC | WDC | Western Digital | — |
| 177 | SP500:NLSN | NLSN | Nielsen Holdings | — |
| 178 | SP500:WU | WU | Western Union | — |
| 179 | SP500:NTRS | NTRS | Northern Trust | — |
| 180 | SP500:TEL | TEL | TE Connectivity | — |
| 181 | SP500:DLR | DLR | Digital Realty Trust | — |
| 182 | SP500:JKHY | JKHY | Jack Henry & Associates | — |
| 183 | SP500:K | K | Kellogg's | — |
| 184 | SP500:EFX | EFX | Equifax | — |
| 185 | SP500:APA | APA | APA Corporation | — |
| 186 | SP500:CAT | CAT | Caterpillar | — |
| 187 | SP500:HIG | HIG | The Hartford | — |
| 188 | SP500:UHS | UHS | Universal Health Services | — |
| 189 | SP500:LH | LH | LabCorp | — |
| 190 | SP500:RMD | RMD | ResMed | — |
| 191 | SP500:TER | TER | Teradyne | — |
| 192 | SP500:DTE | DTE | DTE Energy | — |
| 193 | SP500:BR | BR | Broadridge Financial Solutions | — |
| 194 | SP500:GOOG | GOOG | Alphabet (Class C) | — |
| 195 | SP500:RF | RF | Regions Financial Corporation | — |
| 196 | SP500:BK | BK | BNY Mellon | — |
| 197 | SP500:PPG | PPG | PPG Industries | — |
| 198 | SP500:NEM | NEM | Newmont | — |
| 199 | SP500:UDR | UDR | UDR | — |
| 200 | SP500:WM | WM | Waste Management | — |
| 201 | SP500:JNPR | JNPR | Juniper Networks | — |
| 202 | SP500:PFG | PFG | Principal Financial Group | — |
| 203 | SP500:DPZ | DPZ | Domino's Pizza | — |
| 204 | SP500:BBY | BBY | Best Buy | — |
| 205 | SP500:GPC | GPC | Genuine Parts | — |
| 206 | SP500:ZBRA | ZBRA | Zebra Technologies | — |
| 207 | SP500:MOS | MOS | The Mosaic Company | — |
| 208 | SP500:AAP | AAP | Advance Auto Parts | — |
| 209 | SP500:MGM | MGM | MGM Resorts International | — |
| 210 | SP500:XRAY | XRAY | Dentsply Sirona | — |
| 211 | SP500:ULTA | ULTA | Ulta Beauty | — |
| 212 | SP500:BRO | BRO | Brown & Brown | — |
| 213 | SP500:BLL | BLL | Ball Corp | — |
| 214 | SP500:SHW | SHW | Sherwin-Williams | — |
| 215 | SP500:EXPD | EXPD | Expeditors | — |
| 216 | SP500:APTV | APTV | Aptiv | — |
| 217 | SP500:URI | URI | United Rentals | — |
| 218 | SP500:AMCR | AMCR | Amcor | — |
| 219 | SP500:CCI | CCI | Crown Castle | — |
| 220 | SP500:AKAM | AKAM | Akamai Technologies | — |
| 221 | SP500:AMP | AMP | Ameriprise Financial | — |
| 222 | SP500:PNR | PNR | Pentair | — |
| 223 | SP500:VRSN | VRSN | Verisign | — |
| 224 | SP500:GM | GM | General Motors | — |
| 225 | SP500:HAL | HAL | Halliburton | — |
| 226 | SP500:SNA | SNA | Snap-on | — |
| 227 | SP500:PXD | PXD | Pioneer Natural Resources | — |
| 228 | SP500:AMT | AMT | American Tower | — |
| 229 | SP500:TYL | TYL | Tyler Technologies | — |
| 230 | SP500:HBAN | HBAN | Huntington Bancshares | — |
| 231 | SP500:SWKS | SWKS | Skyworks Solutions | — |
| 232 | SP500:CI | CI | Cigna | — |
| 233 | SP500:XEL | XEL | Xcel Energy | — |
| 234 | SP500:BF-B | BF-B | Brown–Forman | — |
| 235 | SP500:HSIC | HSIC | Henry Schein | — |
| 236 | SP500:ALK | ALK | Alaska Air Group | — |
| 237 | SP500:DOV | DOV | Dover Corporation | — |
| 238 | SP500:UAA | UAA | Under Armour (Class A) | — |
| 239 | SP500:PG | PG | Procter & Gamble | — |
| 240 | SP500:SJM | SJM | JM Smucker | — |
| 241 | SP500:CMA | CMA | Comerica | — |
| 242 | SP500:ADI | ADI | Analog Devices | — |
| 243 | SP500:ECL | ECL | Ecolab | — |
| 244 | SP500:GS | GS | Goldman Sachs | — |
| 245 | SP500:WHR | WHR | Whirlpool Corporation | — |
| 246 | SP500:TRV | TRV | The Travelers Companies | — |
| 247 | SP500:AVY | AVY | Avery Dennison | — |
| 248 | SP500:ADSK | ADSK | Autodesk | — |
| 249 | SP500:UPS | UPS | United Parcel Service | — |
| 250 | SP500:MDLZ | MDLZ | Mondelez International | — |
| 251 | SP500:PWR | PWR | Quanta Services | — |
| 252 | SP500:SBAC | SBAC | SBA Communications | — |
| 253 | SP500:UAL | UAL | United Airlines | — |
| 254 | SP500:CDNS | CDNS | Cadence Design Systems | — |
| 255 | SP500:MAS | MAS | Masco | — |
| 256 | SP500:ABBV | ABBV | AbbVie | — |
| 257 | SP500:BA | BA | Boeing | — |
| 258 | SP500:ROP | ROP | Roper Technologies | — |
| 259 | SP500:LDOS | LDOS | Leidos | — |
| 260 | SP500:IEX | IEX | IDEX Corporation | — |
| 261 | SP500:GPN | GPN | Global Payments | — |
| 262 | SP500:PKI | PKI | PerkinElmer | — |
| 263 | SP500:BWA | BWA | BorgWarner | — |
| 264 | SP500:YUM | YUM | Yum! Brands | — |
| 265 | SP500:ES | ES | Eversource Energy | — |
| 266 | SP500:ZTS | ZTS | Zoetis | — |
| 267 | SP500:MKTX | MKTX | MarketAxess | — |
| 268 | SP500:UNP | UNP | Union Pacific | — |
| 269 | SP500:FCX | FCX | Freeport-McMoRan | — |
| 270 | SP500:RCL | RCL | Royal Caribbean Group | — |
| 271 | SP500:CMI | CMI | Cummins | — |
| 272 | SP500:JCI | JCI | Johnson Controls | — |
| 273 | SP500:NWSA | NWSA | News Corp (Class A) | — |
| 274 | SP500:PFE | PFE | Pfizer | — |
| 275 | SP500:EXPE | EXPE | Expedia Group | — |
| 276 | SP500:FRC | FRC | First Republic Bank | — |
| 277 | SP500:FIS | FIS | Fidelity National Information Services | — |
| 278 | SP500:EMR | EMR | Emerson Electric Company | — |
| 279 | SP500:GLW | GLW | Corning | — |
| 280 | SP500:CNP | CNP | CenterPoint Energy | — |
| 281 | SP500:OKE | OKE | Oneok | — |
| 282 | SP500:KIM | KIM | Kimco Realty | — |
| 283 | SP500:FBHS | FBHS | Fortune Brands Home & Security | — |
| 284 | SP500:DISH | DISH | Dish Network | — |
| 285 | SP500:AES | AES | AES Corp | — |
| 286 | SP500:HON | HON | Honeywell | — |
| 287 | SP500:EVRG | EVRG | Evergy | — |
| 288 | SP500:MTCH | MTCH | Match Group | — |
| 289 | SP500:TFX | TFX | Teleflex | — |
| 290 | SP500:GILD | GILD | Gilead Sciences | — |
| 291 | SP500:CHRW | CHRW | C. H. Robinson | — |
| 292 | SP500:MO | MO | Altria Group | — |
| 293 | SP500:REG | REG | Regency Centers | — |
| 294 | SP500:MMM | MMM | 3M | — |
| 295 | SP500:GIS | GIS | General Mills | — |
| 296 | SP500:INTU | INTU | Intuit | — |
| 297 | SP500:ROK | ROK | Rockwell Automation | — |
| 298 | SP500:PAYC | PAYC | Paycom | — |
| 299 | SP500:HOLX | HOLX | Hologic | — |
| 300 | SP500:LHX | LHX | L3Harris Technologies | — |
| 301 | SP500:COP | COP | ConocoPhillips | — |
| 302 | SP500:AME | AME | Ametek | — |
| 303 | SP500:MCD | MCD | McDonald's | — |
| 304 | SP500:PNW | PNW | Pinnacle West Capital | — |
| 305 | SP500:COO | COO | The Cooper Companies | — |
| 306 | SP500:ILMN | ILMN | Illumina | — |
| 307 | SP500:PEG | PEG | Public Service Enterprise Group | — |
| 308 | SP500:GWW | GWW | W. W. Grainger | — |
| 309 | SP500:O | O | Realty Income Corporation | — |
| 310 | SP500:HAS | HAS | Hasbro | — |
| 311 | SP500:ANSS | ANSS | Ansys | — |
| 312 | SP500:GE | GE | General Electric | — |
| 313 | SP500:RL | RL | Ralph Lauren Corporation | — |
| 314 | SP500:VZ | VZ | Verizon Communications | — |
| 315 | SP500:CSCO | CSCO | Cisco Systems | — |
| 316 | SP500:CMG | CMG | Chipotle Mexican Grill | — |
| 317 | SP500:AZO | AZO | AutoZone | — |
| 318 | SP500:VTR | VTR | Ventas | — |
| 319 | SP500:EBAY | EBAY | eBay | — |
| 320 | SP500:APH | APH | Amphenol | — |
| 321 | SP500:BIO | BIO | Bio-Rad Laboratories | — |
| 322 | SP500:CPRT | CPRT | Copart | — |
| 323 | SP500:STE | STE | Steris | — |
| 324 | SP500:DISCA | DISCA | Discovery (Series A) | — |
| 325 | SP500:TAP | TAP | Molson Coors Beverage Company | — |
| 326 | SP500:MPC | MPC | Marathon Petroleum | — |
| 327 | SP500:MPWR | MPWR | Monolithic Power Systems | — |
| 328 | SP500:NDAQ | NDAQ | Nasdaq | — |
| 329 | SP500:NI | NI | NiSource | — |
| 330 | SP500:ORCL | ORCL | Oracle | — |
| 331 | SP500:CSX | CSX | CSX | — |
| 332 | SP500:PBCT | PBCT | People's United Financial | — |
| 333 | SP500:PM | PM | Philip Morris International | — |
| 334 | SP500:ANTM | ANTM | Anthem | — |
| 335 | SP500:VRSK | VRSK | Verisk Analytics | — |
| 336 | SP500:DFS | DFS | Discover Financial Services | — |
| 337 | SP500:CHD | CHD | Church & Dwight | — |
| 338 | SP500:DG | DG | Dollar General | — |
| 339 | SP500:MCHP | MCHP | Microchip Technology | — |
| 340 | SP500:TMO | TMO | Thermo Fisher Scientific | — |
| 341 | SP500:EMN | EMN | Eastman Chemical | — |
| 342 | SP500:RJF | RJF | Raymond James Financial | — |
| 343 | SP500:JNJ | JNJ | Johnson & Johnson | — |
| 344 | SP500:AWK | AWK | American Water Works | — |
| 345 | SP500:BMY | BMY | Bristol Myers Squibb | — |
| 346 | SP500:CVS | CVS | CVS Health | — |
| 347 | SP500:IVZ | IVZ | Invesco | — |
| 348 | SP500:ROST | ROST | Ross Stores | — |
| 349 | SP500:NLOK | NLOK | NortonLifeLock | — |
| 350 | SP500:XOM | XOM | ExxonMobil | — |
| 351 | SP500:AMD | AMD | Advanced Micro Devices | — |
| 352 | SP500:IPGP | IPGP | IPG Photonics | — |
| 353 | SP500:ATVI | ATVI | Activision Blizzard | — |
| 354 | SP500:HST | HST | Host Hotels & Resorts | — |
| 355 | SP500:TPR | TPR | Tapestry | — |
| 356 | SP500:TECH | TECH | Bio-Techne | — |
| 357 | SP500:WEC | WEC | WEC Energy Group | — |
| 358 | SP500:NVDA | NVDA | Nvidia | — |
| 359 | SP500:CLX | CLX | Clorox | — |
| 360 | SP500:CHTR | CHTR | Charter Communications | — |
| 361 | SP500:HRL | HRL | Hormel | — |
| 362 | SP500:L | L | Loews Corporation | — |
| 363 | SP500:FRT | FRT | Federal Realty Investment Trust | — |
| 364 | SP500:BAC | BAC | Bank of America | — |
| 365 | SP500:CRM | CRM | Salesforce | — |
| 366 | SP500:ALLE | ALLE | Allegion | — |
| 367 | SP500:WST | WST | West Pharmaceutical Services | — |
| 368 | SP500:CTXS | CTXS | Citrix Systems | — |
| 369 | SP500:EA | EA | Electronic Arts | — |
| 370 | SP500:ENPH | ENPH | Enphase Energy | — |
| 371 | SP500:WELL | WELL | Welltower | — |
| 372 | SP500:BLK | BLK | BlackRock | — |
| 373 | SP500:V | V | Visa | — |
| 374 | SP500:ZION | ZION | Zions Bancorp | — |
| 375 | SP500:CE | CE | Celanese | — |
| 376 | SP500:BXP | BXP | Boston Properties | — |
| 377 | SP500:NWL | NWL | Newell Brands | — |
| 378 | SP500:WBA | WBA | Walgreens Boots Alliance | — |
| 379 | SP500:GNRC | GNRC | Generac Holdings | — |
| 380 | SP500:WMT | WMT | Walmart | — |
| 381 | SP500:FLT | FLT | Fleetcor | — |
| 382 | SP500:SPG | SPG | Simon Property Group | — |
| 383 | SP500:ALB | ALB | Albemarle Corporation | — |
| 384 | SP500:WAT | WAT | Waters Corporation | — |
| 385 | SP500:HBI | HBI | Hanesbrands | — |
| 386 | SP500:DXCM | DXCM | DexCom | — |
| 387 | SP500:EIX | EIX | Edison International | — |
| 388 | SP500:DISCK | DISCK | Discovery (Series C) | — |
| 389 | SP500:CAH | CAH | Cardinal Health | — |
| 390 | SP500:VLO | VLO | Valero Energy | — |
| 391 | SP500:HES | HES | Hess Corporation | — |
| 392 | SP500:DD | DD | DuPont | — |
| 393 | SP500:FMC | FMC | FMC Corporation | — |
| 394 | SP500:LYB | LYB | LyondellBasell | — |
| 395 | SP500:SYK | SYK | Stryker Corporation | — |
| 396 | SP500:WMB | WMB | Williams Companies | — |
| 397 | SP500:FAST | FAST | Fastenal | — |
| 398 | SP500:MS | MS | Morgan Stanley | — |
| 399 | SP500:TROW | TROW | T. Rowe Price | — |
| 400 | SP500:SWK | SWK | Stanley Black & Decker | — |
| 401 | SP500:ABMD | ABMD | Abiomed | — |
| 402 | SP500:CRL | CRL | Charles River Laboratories | — |
| 403 | SP500:VRTX | VRTX | Vertex Pharmaceuticals | — |
| 404 | SP500:TSLA | TSLA | Tesla | — |
| 405 | SP500:CB | CB | Chubb | — |
| 406 | SP500:VNO | VNO | Vornado Realty Trust | — |
| 407 | SP500:T | T | AT&T | — |
| 408 | SP500:INCY | INCY | Incyte | — |
| 409 | SP500:EW | EW | Edwards Lifesciences | — |
| 410 | SP500:MHK | MHK | Mohawk Industries | — |
| 411 | SP500:CCL | CCL | Carnival Corporation | — |
| 412 | SP500:ATO | ATO | Atmos Energy | — |
| 413 | SP500:IDXX | IDXX | Idexx Laboratories | — |
| 414 | SP500:DRI | DRI | Darden Restaurants | — |
| 415 | SP500:NUE | NUE | Nucor | — |
| 416 | SP500:SYY | SYY | Sysco | — |
| 417 | SP500:MNST | MNST | Monster Beverage | — |
| 418 | SP500:STZ | STZ | Constellation Brands | — |
| 419 | SP500:FANG | FANG | Diamondback Energy | — |
| 420 | SP500:RHI | RHI | Robert Half International | — |
| 421 | SP500:DHR | DHR | Danaher Corporation | — |
| 422 | SP500:ICE | ICE | Intercontinental Exchange | — |
| 423 | SP500:AAPL | AAPL | Apple | — |
| 424 | SP500:CTAS | CTAS | Cintas Corporation | — |
| 425 | SP500:D | D | Dominion Energy | — |
| 426 | SP500:ACN | ACN | Accenture | — |
| 427 | SP500:IBM | IBM | IBM | — |
| 428 | SP500:INTC | INTC | Intel | — |
| 429 | SP500:BKNG | BKNG | Booking Holdings | — |
| 430 | SP500:COST | COST | Costco | — |
| 431 | SP500:AON | AON | Aon | — |
| 432 | SP500:STX | STX | Seagate Technology | — |
| 433 | SP500:ED | ED | Consolidated Edison | — |
| 434 | SP500:LEN | LEN | Lennar | — |
| 435 | SP500:NFLX | NFLX | Netflix | — |
| 436 | SP500:HII | HII | Huntington Ingalls Industries | — |
| 437 | SP500:ALL | ALL | Allstate Corp | — |
| 438 | SP500:ITW | ITW | Illinois Tool Works | — |
| 439 | SP500:CDW | CDW | CDW | — |
| 440 | SP500:BSX | BSX | Boston Scientific | — |
| 441 | SP500:CNC | CNC | Centene Corporation | — |
| 442 | SP500:ALGN | ALGN | Align Technology | — |
| 443 | SP500:SLB | SLB | Schlumberger | — |
| 444 | SP500:RE | RE | Everest Re | — |
| 445 | SP500:TTWO | TTWO | Take-Two Interactive | — |
| 446 | SP500:SIVB | SIVB | SVB Financial | — |
| 447 | SP500:QCOM | QCOM | Qualcomm | — |
| 448 | SP500:MTB | MTB | M&T Bank | — |
| 449 | SP500:AVB | AVB | AvalonBay Communities | — |
| 450 | SP500:NVR | NVR | NVR | — |
| 451 | SP500:LYV | LYV | Live Nation Entertainment | — |
| 452 | SP500:LVS | LVS | Las Vegas Sands | — |
| 453 | SP500:LNC | LNC | Lincoln National | — |
| 454 | SP500:BAX | BAX | Baxter International | — |
| 455 | SP500:SNPS | SNPS | Synopsys | — |
| 456 | SP500:PPL | PPL | PPL | — |
| 457 | SP500:TSCO | TSCO | Tractor Supply Company | — |
| 458 | SP500:LNT | LNT | Alliant Energy | — |
| 459 | SP500:CTSH | CTSH | Cognizant Technology Solutions | — |
| 460 | SP500:CERN | CERN | Cerner | — |
| 461 | SP500:TFC | TFC | Truist Financial | — |
| 462 | SP500:BEN | BEN | Franklin Resources | — |
| 463 | SP500:LIN | LIN | Linde | — |
| 464 | SP500:AIG | AIG | American International Group | — |
| 465 | SP500:AEP | AEP | American Electric Power | — |
| 466 | SP500:FB | FB | Facebook | — |
| 467 | SP500:CVX | CVX | Chevron Corporation | — |
| 468 | SP500:SPGI | SPGI | S&P Global | — |
| 469 | SP500:J | J | Jacobs Engineering Group | — |
| 470 | SP500:MAR | MAR | Marriott International | — |
| 471 | SP500:HLT | HLT | Hilton Worldwide | — |
| 472 | SP500:MU | MU | Micron Technology | — |
| 473 | SP500:RTX | RTX | Raytheon Technologies | — |

### NYSE：1737 只

| array_index | 建议 stock_uid | 股票代码 ticker | 公司名称（本地快照） | 源 Wikidata QID |
|---:|---|---|---|---|
| 0 | NYSE:A | A | Agilent Technologies | Q393762 |
| 1 | NYSE:AAN | AAN | 未提供 | Q10397787 |
| 2 | NYSE:AAP | AAP | Advance Auto Parts | Q4686051 |
| 3 | NYSE:AAT | AAT | 未提供 | — |
| 4 | NYSE:AB | AB | 未提供 | Q4732268 |
| 5 | NYSE:ABB | ABB | 未提供 | Q52825 |
| 6 | NYSE:ABBV | ABBV | AbbVie | Q14662364 |
| 7 | NYSE:ABC | ABC | AmerisourceBergen | Q470156 |
| 8 | NYSE:ABG | ABG | 未提供 | Q4803735 |
| 9 | NYSE:ABM | ABM | 未提供 | Q4650338 |
| 10 | NYSE:ABR | ABR | 未提供 | — |
| 11 | NYSE:ABT | ABT | Abbott Laboratories | Q306764 |
| 12 | NYSE:ABX | ABX | 未提供 | Q808908 |
| 13 | NYSE:ACC | ACC | 未提供 | Q4743275 |
| 14 | NYSE:ACCO | ACCO | 未提供 | Q288129 |
| 15 | NYSE:ACH | ACH | 未提供 | Q447855 |
| 16 | NYSE:ACM | ACM | 未提供 | Q2825507 |
| 17 | NYSE:ACN | ACN | Accenture | Q338825 |
| 18 | NYSE:ACP | ACP | 未提供 | — |
| 19 | NYSE:ACRE | ACRE | 未提供 | — |
| 20 | NYSE:ADC | ADC | 未提供 | — |
| 21 | NYSE:ADM | ADM | ADM | Q632694 |
| 22 | NYSE:ADS | ADS | 未提供 | — |
| 23 | NYSE:ADX | ADX | 未提供 | Q4680345 |
| 24 | NYSE:AEB | AEB | 未提供 | — |
| 25 | NYSE:AED | AED | 未提供 | — |
| 26 | NYSE:AEE | AEE | Ameren Corp | Q462984 |
| 27 | NYSE:AEH | AEH | 未提供 | — |
| 28 | NYSE:AEL | AEL | 未提供 | — |
| 29 | NYSE:AEM | AEM | 未提供 | — |
| 30 | NYSE:AEO | AEO | 未提供 | Q2842931 |
| 31 | NYSE:AEP | AEP | American Electric Power | Q464092 |
| 32 | NYSE:AER | AER | 未提供 | Q4687934 |
| 33 | NYSE:AES | AES | AES Corp | Q291508 |
| 34 | NYSE:AET | AET | 未提供 | Q26049 |
| 35 | NYSE:AFB | AFB | 未提供 | — |
| 36 | NYSE:AFC | AFC | 未提供 | — |
| 37 | NYSE:AFG | AFG | 未提供 | Q4743779 |
| 38 | NYSE:AFL | AFL | Aflac | Q26311 |
| 39 | NYSE:AFT | AFT | 未提供 | — |
| 40 | NYSE:AGCO | AGCO | 未提供 | Q292292 |
| 41 | NYSE:AGD | AGD | 未提供 | — |
| 42 | NYSE:AGM | AGM | 未提供 | Q5439999 |
| 43 | NYSE:AGO | AGO | 未提供 | — |
| 44 | NYSE:AGO-B | AGO-B | 未提供 | — |
| 45 | NYSE:AGO-E | AGO-E | 未提供 | — |
| 46 | NYSE:AGO-F | AGO-F | 未提供 | — |
| 47 | NYSE:AGRO | AGRO | 未提供 | — |
| 48 | NYSE:AGU | AGU | 未提供 | — |
| 49 | NYSE:AGX | AGX | 未提供 | — |
| 50 | NYSE:AHL | AHL | 未提供 | — |
| 51 | NYSE:AHT-D | AHT-D | 未提供 | — |
| 52 | NYSE:AI | AI | 未提供 | — |
| 53 | NYSE:AIG | AIG | American International Group | Q212235 |
| 54 | NYSE:AIN | AIN | 未提供 | — |
| 55 | NYSE:AIR | AIR | 未提供 | Q4649932 |
| 56 | NYSE:AIT | AIT | 未提供 | Q4781544 |
| 57 | NYSE:AIV | AIV | 未提供 | Q293796 |
| 58 | NYSE:AIZ | AIZ | Assurant | — |
| 59 | NYSE:AJG | AJG | Arthur J. Gallagher & Co. | — |
| 60 | NYSE:AKO.A | AKO.A | 未提供 | — |
| 61 | NYSE:AKO.B | AKO.B | 未提供 | — |
| 62 | NYSE:AKP | AKP | 未提供 | — |
| 63 | NYSE:AKR | AKR | 未提供 | Q4671583 |
| 64 | NYSE:AL | AL | 未提供 | Q4698062 |
| 65 | NYSE:ALB | ALB | Albemarle Corporation | Q127074 |
| 66 | NYSE:ALDW | ALDW | 未提供 | — |
| 67 | NYSE:ALE | ALE | 未提供 | — |
| 68 | NYSE:ALEX | ALEX | 未提供 | — |
| 69 | NYSE:ALG | ALG | 未提供 | Q21198484 |
| 70 | NYSE:ALK | ALK | Alaska Air Group | Q822614 |
| 71 | NYSE:ALL | ALL | Allstate Corp | — |
| 72 | NYSE:ALSN | ALSN | 未提供 | — |
| 73 | NYSE:ALV | ALV | 未提供 | Q787044 |
| 74 | NYSE:ALX | ALX | 未提供 | — |
| 75 | NYSE:AME | AME | Ametek | Q470317 |
| 76 | NYSE:AMG | AMG | 未提供 | — |
| 77 | NYSE:AMN | AMN | 未提供 | — |
| 78 | NYSE:AMOV | AMOV | 未提供 | — |
| 79 | NYSE:AMP | AMP | Ameriprise Financial | Q2843129 |
| 80 | NYSE:AMX | AMX | 未提供 | — |
| 81 | NYSE:AN | AN | 未提供 | Q784804 |
| 82 | NYSE:ANDV | ANDV | 未提供 | — |
| 83 | NYSE:ANDX | ANDX | 未提供 | — |
| 84 | NYSE:ANF | ANF | 未提供 | Q319344 |
| 85 | NYSE:ANH-A | ANH-A | 未提供 | — |
| 86 | NYSE:ANTM | ANTM | Anthem | Q677488 |
| 87 | NYSE:AOD | AOD | 未提供 | — |
| 88 | NYSE:AON | AON | Aon | Q739518 |
| 89 | NYSE:AOS | AOS | A. O. Smith | — |
| 90 | NYSE:AP | AP | 未提供 | Q4748076 |
| 91 | NYSE:APA | APA | APA Corporation | Q119991 |
| 92 | NYSE:APB | APB | 未提供 | — |
| 93 | NYSE:APC | APC | 未提供 | Q483637 |
| 94 | NYSE:APD | APD | Air Products & Chemicals | Q407744 |
| 95 | NYSE:APF | APF | 未提供 | — |
| 96 | NYSE:APO | APO | 未提供 | Q619121 |
| 97 | NYSE:APTS | APTS | 未提供 | — |
| 98 | NYSE:APTV | APTV | Aptiv | — |
| 99 | NYSE:APU | APU | 未提供 | — |
| 100 | NYSE:ARDC | ARDC | 未提供 | — |
| 101 | NYSE:ARE | ARE | Alexandria Real Estate Equities | — |
| 102 | NYSE:ARI | ARI | 未提供 | — |
| 103 | NYSE:ARNC | ARNC | 未提供 | Q25104597 |
| 104 | NYSE:ARR | ARR | 未提供 | — |
| 105 | NYSE:ARR-A | ARR-A | 未提供 | — |
| 106 | NYSE:ARW | ARW | 未提供 | Q2059537 |
| 107 | NYSE:ASA | ASA | 未提供 | — |
| 108 | NYSE:ASB | ASB | 未提供 | — |
| 109 | NYSE:ASGN | ASGN | 未提供 | — |
| 110 | NYSE:ASR | ASR | 未提供 | — |
| 111 | NYSE:ATGE | ATGE | 未提供 | — |
| 112 | NYSE:ATI | ATI | 未提供 | Q1361648 |
| 113 | NYSE:ATO | ATO | Atmos Energy | — |
| 114 | NYSE:ATR | ATR | 未提供 | — |
| 115 | NYSE:ATU | ATU | 未提供 | — |
| 116 | NYSE:AU | AU | 未提供 | Q541076 |
| 117 | NYSE:AVA | AVA | 未提供 | Q4828988 |
| 118 | NYSE:AVB | AVB | AvalonBay Communities | Q4827537 |
| 119 | NYSE:AVD | AVD | 未提供 | — |
| 120 | NYSE:AVK | AVK | 未提供 | — |
| 121 | NYSE:AVT | AVT | 未提供 | Q4829076 |
| 122 | NYSE:AVX | AVX | 未提供 | — |
| 123 | NYSE:AVY | AVY | Avery Dennison | Q790650 |
| 124 | NYSE:AWF | AWF | 未提供 | — |
| 125 | NYSE:AWI | AWI | 未提供 | Q689446 |
| 126 | NYSE:AWK | AWK | American Water Works | Q467238 |
| 127 | NYSE:AWR | AWR | 未提供 | — |
| 128 | NYSE:AXE | AXE | 未提供 | — |
| 129 | NYSE:AXL | AXL | 未提供 | Q4743120 |
| 130 | NYSE:AXP | AXP | American Express | Q194360 |
| 131 | NYSE:AXS | AXS | 未提供 | — |
| 132 | NYSE:AYI | AYI | 未提供 | Q4677870 |
| 133 | NYSE:AYR | AYR | 未提供 | Q13583352 |
| 134 | NYSE:AZN | AZN | 未提供 | Q731938 |
| 135 | NYSE:AZO | AZO | AutoZone | Q4826087 |
| 136 | NYSE:AZZ | AZZ | 未提供 | — |
| 137 | NYSE:B | B | 未提供 | Q4861529 |
| 138 | NYSE:BA | BA | Boeing | Q66 |
| 139 | NYSE:BAC | BAC | Bank of America | Q487907 |
| 140 | NYSE:BAC-D | BAC-D | 未提供 | — |
| 141 | NYSE:BAC-E | BAC-E | 未提供 | — |
| 142 | NYSE:BAC-I | BAC-I | 未提供 | — |
| 143 | NYSE:BAC-L | BAC-L | 未提供 | — |
| 144 | NYSE:BAF | BAF | 未提供 | — |
| 145 | NYSE:BAH | BAH | 未提供 | Q893221 |
| 146 | NYSE:BAK | BAK | 未提供 | — |
| 147 | NYSE:BAM | BAM | 未提供 | Q908324 |
| 148 | NYSE:BAP | BAP | 未提供 | — |
| 149 | NYSE:BAX | BAX | Baxter International | — |
| 150 | NYSE:BB | BB | 未提供 | — |
| 151 | NYSE:BBF | BBF | 未提供 | — |
| 152 | NYSE:BBK | BBK | 未提供 | — |
| 153 | NYSE:BBL | BBL | 未提供 | Q625713 |
| 154 | NYSE:BBN | BBN | 未提供 | — |
| 155 | NYSE:BBT | BBT | 未提供 | Q795486 |
| 156 | NYSE:BBT-D | BBT-D | 未提供 | — |
| 157 | NYSE:BBT-E | BBT-E | 未提供 | — |
| 158 | NYSE:BBT-F | BBT-F | 未提供 | — |
| 159 | NYSE:BBVA | BBVA | 未提供 | Q806189 |
| 160 | NYSE:BBY | BBY | Best Buy | Q533415 |
| 161 | NYSE:BC | BC | 未提供 | Q994153 |
| 162 | NYSE:BCE | BCE | 未提供 | Q815694 |
| 163 | NYSE:BCH | BCH | 未提供 | — |
| 164 | NYSE:BCO | BCO | 未提供 | Q2902305 |
| 165 | NYSE:BCR | BCR | 未提供 | Q5006774 |
| 166 | NYSE:BCS | BCS | 未提供 | Q245343 |
| 167 | NYSE:BCS-D | BCS-D | 未提供 | — |
| 168 | NYSE:BCX | BCX | 未提供 | — |
| 169 | NYSE:BDC | BDC | 未提供 | — |
| 170 | NYSE:BDJ | BDJ | 未提供 | — |
| 171 | NYSE:BDN | BDN | 未提供 | Q4957288 |
| 172 | NYSE:BDX | BDX | Becton Dickinson | Q813780 |
| 173 | NYSE:BEL | BEL | 未提供 | Q4045977 |
| 174 | NYSE:BEN | BEN | Franklin Resources | Q744683 |
| 175 | NYSE:BEP | BEP | 未提供 | — |
| 176 | NYSE:BERY | BERY | 未提供 | — |
| 177 | NYSE:BF.A | BF.A | 未提供 | — |
| 178 | NYSE:BF.B | BF.B | 未提供 | — |
| 179 | NYSE:BFK | BFK | 未提供 | — |
| 180 | NYSE:BFO | BFO | 未提供 | — |
| 181 | NYSE:BFS | BFS | 未提供 | — |
| 182 | NYSE:BFZ | BFZ | 未提供 | — |
| 183 | NYSE:BG | BG | 未提供 | Q1009458 |
| 184 | NYSE:BGB | BGB | 未提供 | — |
| 185 | NYSE:BGC | BGC | 未提供 | — |
| 186 | NYSE:BGG | BGG | 未提供 | Q915543 |
| 187 | NYSE:BGH | BGH | 未提供 | — |
| 188 | NYSE:BGR | BGR | 未提供 | — |
| 189 | NYSE:BGS | BGS | 未提供 | — |
| 190 | NYSE:BGT | BGT | 未提供 | — |
| 191 | NYSE:BGX | BGX | 未提供 | — |
| 192 | NYSE:BGY | BGY | 未提供 | — |
| 193 | NYSE:BH | BH | 未提供 | Q4906978 |
| 194 | NYSE:BHE | BHE | 未提供 | Q4886950 |
| 195 | NYSE:BHK | BHK | 未提供 | — |
| 196 | NYSE:BHLB | BHLB | 未提供 | — |
| 197 | NYSE:BHP | BHP | 未提供 | — |
| 198 | NYSE:BID | BID | 未提供 | Q654762 |
| 199 | NYSE:BIF | BIF | 未提供 | — |
| 200 | NYSE:BIG | BIG | 未提供 | Q4905973 |
| 201 | NYSE:BIO | BIO | Bio-Rad Laboratories | Q2904022 |
| 202 | NYSE:BIO.B | BIO.B | 未提供 | — |
| 203 | NYSE:BIP | BIP | 未提供 | — |
| 204 | NYSE:BITA | BITA | 未提供 | — |
| 205 | NYSE:BJZ | BJZ | 未提供 | — |
| 206 | NYSE:BK | BK | BNY Mellon | — |
| 207 | NYSE:BK-C | BK-C | 未提供 | — |
| 208 | NYSE:BKD | BKD | 未提供 | — |
| 209 | NYSE:BKE | BKE | 未提供 | Q4983306 |
| 210 | NYSE:BKH | BKH | 未提供 | — |
| 211 | NYSE:BKK | BKK | 未提供 | — |
| 212 | NYSE:BKN | BKN | 未提供 | — |
| 213 | NYSE:BKS | BKS | 未提供 | Q795454 |
| 214 | NYSE:BKT | BKT | 未提供 | — |
| 215 | NYSE:BKU | BKU | 未提供 | — |
| 216 | NYSE:BLH | BLH | 未提供 | — |
| 217 | NYSE:BLK | BLK | BlackRock | Q219635 |
| 218 | NYSE:BLL | BLL | Ball Corp | Q4034815 |
| 219 | NYSE:BLW | BLW | 未提供 | — |
| 220 | NYSE:BLX | BLX | 未提供 | — |
| 221 | NYSE:BMA | BMA | 未提供 | Q2335199 |
| 222 | NYSE:BME | BME | 未提供 | — |
| 223 | NYSE:BMI | BMI | 未提供 | — |
| 224 | NYSE:BML-G | BML-G | 未提供 | — |
| 225 | NYSE:BML-H | BML-H | 未提供 | — |
| 226 | NYSE:BML-I | BML-I | 未提供 | — |
| 227 | NYSE:BML-J | BML-J | 未提供 | — |
| 228 | NYSE:BML-L | BML-L | 未提供 | — |
| 229 | NYSE:BMO | BMO | 未提供 | Q806693 |
| 230 | NYSE:BMS | BMS | 未提供 | — |
| 231 | NYSE:BMY | BMY | Bristol Myers Squibb | Q266423 |
| 232 | NYSE:BNJ | BNJ | 未提供 | — |
| 233 | NYSE:BNS | BNS | 未提供 | Q451476 |
| 234 | NYSE:BNY | BNY | 未提供 | — |
| 235 | NYSE:BOE | BOE | 未提供 | — |
| 236 | NYSE:BOH | BOH | 未提供 | Q4856124 |
| 237 | NYSE:BP | BP | 未提供 | Q1049970 |
| 238 | NYSE:BPI | BPI | 未提供 | Q4966553 |
| 239 | NYSE:BPK | BPK | 未提供 | — |
| 240 | NYSE:BPL | BPL | 未提供 | Q4983090 |
| 241 | NYSE:BPT | BPT | 未提供 | Q4836297 |
| 242 | NYSE:BQH | BQH | 未提供 | — |
| 243 | NYSE:BR | BR | Broadridge Financial Solutions | — |
| 244 | NYSE:BRC | BRC | 未提供 | Q897097 |
| 245 | NYSE:BRFS | BRFS | 未提供 | Q899097 |
| 246 | NYSE:BRK.B | BRK.B | 未提供 | — |
| 247 | NYSE:BRO | BRO | Brown & Brown | — |
| 248 | NYSE:BRS | BRS | 未提供 | — |
| 249 | NYSE:BRT | BRT | 未提供 | — |
| 250 | NYSE:BSAC | BSAC | 未提供 | — |
| 251 | NYSE:BSD | BSD | 未提供 | — |
| 252 | NYSE:BSE | BSE | 未提供 | — |
| 253 | NYSE:BSL | BSL | 未提供 | — |
| 254 | NYSE:BSMX | BSMX | 未提供 | — |
| 255 | NYSE:BSX | BSX | Boston Scientific | Q894585 |
| 256 | NYSE:BT | BT | 未提供 | Q593786 |
| 257 | NYSE:BTA | BTA | 未提供 | — |
| 258 | NYSE:BTI | BTI | 未提供 | — |
| 259 | NYSE:BTO | BTO | 未提供 | — |
| 260 | NYSE:BTT | BTT | 未提供 | — |
| 261 | NYSE:BTZ | BTZ | 未提供 | — |
| 262 | NYSE:BUD | BUD | 未提供 | Q128738 |
| 263 | NYSE:BUI | BUI | 未提供 | — |
| 264 | NYSE:BWA | BWA | BorgWarner | Q825943 |
| 265 | NYSE:BWG | BWG | 未提供 | — |
| 266 | NYSE:BWP | BWP | 未提供 | — |
| 267 | NYSE:BWXT | BWXT | 未提供 | — |
| 268 | NYSE:BX | BX | 未提供 | Q880942 |
| 269 | NYSE:BXMX | BXMX | 未提供 | — |
| 270 | NYSE:BXP | BXP | Boston Properties | — |
| 271 | NYSE:BYD | BYD | 未提供 | — |
| 272 | NYSE:BYM | BYM | 未提供 | — |
| 273 | NYSE:BZH | BZH | 未提供 | — |
| 274 | NYSE:C | C | Citigroup | Q219508 |
| 275 | NYSE:C-N | C-N | 未提供 | — |
| 276 | NYSE:C-P | C-P | 未提供 | — |
| 277 | NYSE:CACI | CACI | 未提供 | — |
| 278 | NYSE:CAE | CAE | 未提供 | — |
| 279 | NYSE:CAF | CAF | 未提供 | — |
| 280 | NYSE:CAG | CAG | Conagra Brands | Q1123399 |
| 281 | NYSE:CAH | CAH | Cardinal Health | — |
| 282 | NYSE:CAJ | CAJ | 未提供 | Q62621 |
| 283 | NYSE:CAL | CAL | 未提供 | Q4112833 |
| 284 | NYSE:CAPL | CAPL | 未提供 | — |
| 285 | NYSE:CAT | CAT | Caterpillar | Q459965 |
| 286 | NYSE:CATO | CATO | 未提供 | — |
| 287 | NYSE:CB | CB | Chubb | — |
| 288 | NYSE:CBB | CBB | 未提供 | — |
| 289 | NYSE:CBB-B | CBB-B | 未提供 | — |
| 290 | NYSE:CBD | CBD | 未提供 | — |
| 291 | NYSE:CBG | CBG | 未提供 | Q1023013 |
| 292 | NYSE:CBI | CBI | 未提供 | Q5095516 |
| 293 | NYSE:CBL | CBL | 未提供 | Q5009135 |
| 294 | NYSE:CBL-D | CBL-D | 未提供 | — |
| 295 | NYSE:CBL-E | CBL-E | 未提供 | — |
| 296 | NYSE:CBM | CBM | 未提供 | — |
| 297 | NYSE:CBS | CBS | 未提供 | Q950380 |
| 298 | NYSE:CBS.A | CBS.A | 未提供 | — |
| 299 | NYSE:CBT | CBT | 未提供 | — |
| 300 | NYSE:CBU | CBU | 未提供 | — |
| 301 | NYSE:CBZ | CBZ | 未提供 | — |
| 302 | NYSE:CCC | CCC | 未提供 | — |
| 303 | NYSE:CCJ | CCJ | 未提供 | Q1028449 |
| 304 | NYSE:CCK | CCK | 未提供 | Q5189378 |
| 305 | NYSE:CCL | CCL | Carnival Corporation | Q1044059 |
| 306 | NYSE:CCU | CCU | 未提供 | — |
| 307 | NYSE:CDR-B | CDR-B | 未提供 | — |
| 308 | NYSE:CE | CE | Celanese | — |
| 309 | NYSE:CEA | CEA | 未提供 | Q291796 |
| 310 | NYSE:CEE | CEE | 未提供 | — |
| 311 | NYSE:CEM | CEM | 未提供 | — |
| 312 | NYSE:CEO | CEO | 未提供 | Q795221 |
| 313 | NYSE:CF | CF | CF Industries | Q905806 |
| 314 | NYSE:CFC-B | CFC-B | 未提供 | — |
| 315 | NYSE:CFR | CFR | 未提供 | Q5506152 |
| 316 | NYSE:CFX | CFX | 未提供 | — |
| 317 | NYSE:CHA | CHA | 未提供 | — |
| 318 | NYSE:CHD | CHD | Church & Dwight | — |
| 319 | NYSE:CHE | CHE | 未提供 | — |
| 320 | NYSE:CHH | CHH | 未提供 | Q1075788 |
| 321 | NYSE:CHK-D | CHK-D | 未提供 | — |
| 322 | NYSE:CHL | CHL | 未提供 | Q741618 |
| 323 | NYSE:CHN | CHN | 未提供 | — |
| 324 | NYSE:CHS | CHS | 未提供 | — |
| 325 | NYSE:CHSP | CHSP | 未提供 | — |
| 326 | NYSE:CHT | CHT | 未提供 | — |
| 327 | NYSE:CHU | CHU | 未提供 | Q1068485 |
| 328 | NYSE:CI | CI | Cigna | — |
| 329 | NYSE:CIA | CIA | 未提供 | — |
| 330 | NYSE:CIB | CIB | 未提供 | Q806206 |
| 331 | NYSE:CIEN | CIEN | 未提供 | — |
| 332 | NYSE:CII | CII | 未提供 | — |
| 333 | NYSE:CIM | CIM | 未提供 | — |
| 334 | NYSE:CIR | CIR | 未提供 | — |
| 335 | NYSE:CIT | CIT | 未提供 | — |
| 336 | NYSE:CKH | CKH | 未提供 | — |
| 337 | NYSE:CL | CL | Colgate-Palmolive | Q609466 |
| 338 | NYSE:CLB | CLB | 未提供 | — |
| 339 | NYSE:CLDT | CLDT | 未提供 | — |
| 340 | NYSE:CLGX | CLGX | 未提供 | — |
| 341 | NYSE:CLH | CLH | 未提供 | Q5130494 |
| 342 | NYSE:CLI | CLI | 未提供 | — |
| 343 | NYSE:CLR | CLR | 未提供 | — |
| 344 | NYSE:CLS | CLS | 未提供 | — |
| 345 | NYSE:CLW | CLW | 未提供 | Q5130952 |
| 346 | NYSE:CLX | CLX | Clorox | Q910368 |
| 347 | NYSE:CM | CM | 未提供 | Q666694 |
| 348 | NYSE:CMA | CMA | Comerica | Q1114148 |
| 349 | NYSE:CMC | CMC | 未提供 | — |
| 350 | NYSE:CMD | CMD | 未提供 | — |
| 351 | NYSE:CMG | CMG | Chipotle Mexican Grill | Q465751 |
| 352 | NYSE:CMI | CMI | Cummins | Q173476 |
| 353 | NYSE:CMO | CMO | 未提供 | — |
| 354 | NYSE:CMP | CMP | 未提供 | — |
| 355 | NYSE:CMRE | CMRE | 未提供 | — |
| 356 | NYSE:CMS | CMS | CMS Energy | Q5013012 |
| 357 | NYSE:CNA | CNA | 未提供 | — |
| 358 | NYSE:CNC | CNC | Centene Corporation | Q5059102 |
| 359 | NYSE:CNI | CNI | 未提供 | — |
| 360 | NYSE:CNK | CNK | 未提供 | — |
| 361 | NYSE:CNO | CNO | 未提供 | — |
| 362 | NYSE:CNP | CNP | CenterPoint Energy | — |
| 363 | NYSE:CNQ | CNQ | 未提供 | Q1032229 |
| 364 | NYSE:CNS | CNS | 未提供 | — |
| 365 | NYSE:CODI | CODI | 未提供 | — |
| 366 | NYSE:COF | COF | Capital One Financial | Q1034654 |
| 367 | NYSE:COF-P | COF-P | 未提供 | — |
| 368 | NYSE:COG | COG | 未提供 | — |
| 369 | NYSE:COL | COL | 未提供 | Q461210 |
| 370 | NYSE:COO | COO | The Cooper Companies | — |
| 371 | NYSE:COP | COP | ConocoPhillips | Q214763 |
| 372 | NYSE:COR | COR | 未提供 | — |
| 373 | NYSE:COT | COT | 未提供 | — |
| 374 | NYSE:CP | CP | 未提供 | — |
| 375 | NYSE:CPA | CPA | 未提供 | — |
| 376 | NYSE:CPB | CPB | Campbell Soup | — |
| 377 | NYSE:CPF | CPF | 未提供 | Q5061578 |
| 378 | NYSE:CPG | CPG | 未提供 | — |
| 379 | NYSE:CPK | CPK | 未提供 | — |
| 380 | NYSE:CPL | CPL | 未提供 | Q1121216 |
| 381 | NYSE:CPN | CPN | 未提供 | Q5023914 |
| 382 | NYSE:CPT | CPT | 未提供 | Q5025803 |
| 383 | NYSE:CR | CR | 未提供 | — |
| 384 | NYSE:CRH | CRH | 未提供 | Q1053299 |
| 385 | NYSE:CRI | CRI | 未提供 | — |
| 386 | NYSE:CRL | CRL | Charles River Laboratories | Q5082076 |
| 387 | NYSE:CRM | CRM | Salesforce | Q941127 |
| 388 | NYSE:CRR | CRR | 未提供 | — |
| 389 | NYSE:CRS | CRS | 未提供 | — |
| 390 | NYSE:CRT | CRT | 未提供 | — |
| 391 | NYSE:CRY | CRY | 未提供 | — |
| 392 | NYSE:CS | CS | 未提供 | Q372657 |
| 393 | NYSE:CSL | CSL | 未提供 | Q5041354 |
| 394 | NYSE:CSS | CSS | 未提供 | Q5014304 |
| 395 | NYSE:CSU | CSU | 未提供 | Q5035662 |
| 396 | NYSE:CSV | CSV | 未提供 | — |
| 397 | NYSE:CTB | CTB | 未提供 | Q1129847 |
| 398 | NYSE:CTL | CTL | 未提供 | — |
| 399 | NYSE:CTR | CTR | 未提供 | — |
| 400 | NYSE:CTS | CTS | 未提供 | — |
| 401 | NYSE:CTW | CTW | 未提供 | — |
| 402 | NYSE:CUB | CUB | 未提供 | Q5192235 |
| 403 | NYSE:CUBE | CUBE | 未提供 | Q5192200 |
| 404 | NYSE:CUBI | CUBI | 未提供 | — |
| 405 | NYSE:CUK | CUK | 未提供 | — |
| 406 | NYSE:CULP | CULP | 未提供 | — |
| 407 | NYSE:CUZ | CUZ | 未提供 | Q5178840 |
| 408 | NYSE:CVA | CVA | 未提供 | — |
| 409 | NYSE:CVE | CVE | 未提供 | — |
| 410 | NYSE:CVG | CVG | 未提供 | Q1129426 |
| 411 | NYSE:CVI | CVI | 未提供 | — |
| 412 | NYSE:CVS | CVS | CVS Health | Q624375 |
| 413 | NYSE:CVX | CVX | Chevron Corporation | Q319642 |
| 414 | NYSE:CW | CW | 未提供 | Q1138291 |
| 415 | NYSE:CWT | CWT | 未提供 | — |
| 416 | NYSE:CXH | CXH | 未提供 | — |
| 417 | NYSE:CXO | CXO | 未提供 | — |
| 418 | NYSE:CXW | CXW | 未提供 | Q1135278 |
| 419 | NYSE:CYD | CYD | 未提供 | — |
| 420 | NYSE:CYS | CYS | 未提供 | — |
| 421 | NYSE:CYS-A | CYS-A | 未提供 | — |
| 422 | NYSE:D | D | Dominion Energy | Q677464 |
| 423 | NYSE:DAL | DAL | Delta Air Lines | Q188920 |
| 424 | NYSE:DAN | DAN | 未提供 | Q5214701 |
| 425 | NYSE:DAR | DAR | 未提供 | — |
| 426 | NYSE:DB | DB | 未提供 | Q66048 |
| 427 | NYSE:DBD | DBD | 未提供 | — |
| 428 | NYSE:DBL | DBL | 未提供 | — |
| 429 | NYSE:DCI | DCI | 未提供 | Q5295362 |
| 430 | NYSE:DCM | DCM | 未提供 | Q853958 |
| 431 | NYSE:DCO | DCO | 未提供 | — |
| 432 | NYSE:DCP | DCP | 未提供 | — |
| 433 | NYSE:DDD | DDD | 未提供 | — |
| 434 | NYSE:DDF | DDF | 未提供 | — |
| 435 | NYSE:DDR | DDR | 未提供 | Q1206686 |
| 436 | NYSE:DDR-J | DDR-J | 未提供 | — |
| 437 | NYSE:DDS | DDS | 未提供 | Q844805 |
| 438 | NYSE:DDT | DDT | 未提供 | — |
| 439 | NYSE:DE | DE | Deere & Co. | Q496302 |
| 440 | NYSE:DECK | DECK | 未提供 | — |
| 441 | NYSE:DEI | DEI | 未提供 | — |
| 442 | NYSE:DEL | DEL | 未提供 | — |
| 443 | NYSE:DEO | DEO | 未提供 | Q161140 |
| 444 | NYSE:DEX | DEX | 未提供 | — |
| 445 | NYSE:DFS | DFS | Discover Financial Services | Q5281775 |
| 446 | NYSE:DG | DG | Dollar General | — |
| 447 | NYSE:DGX | DGX | Quest Diagnostics | Q7271456 |
| 448 | NYSE:DHG | DHG | 未提供 | — |
| 449 | NYSE:DHI | DHI | D. R. Horton | Q5203782 |
| 450 | NYSE:DHR | DHR | Danaher Corporation | Q1159256 |
| 451 | NYSE:DIN | DIN | 未提供 | Q5278020 |
| 452 | NYSE:DIS | DIS | The Walt Disney Company | Q7414 |
| 453 | NYSE:DKL | DKL | 未提供 | — |
| 454 | NYSE:DKS | DKS | 未提供 | Q5272601 |
| 455 | NYSE:DKT | DKT | 未提供 | — |
| 456 | NYSE:DLB | DLB | 未提供 | — |
| 457 | NYSE:DLR | DLR | Digital Realty Trust | — |
| 458 | NYSE:DLX | DLX | 未提供 | Q5254919 |
| 459 | NYSE:DMO | DMO | 未提供 | — |
| 460 | NYSE:DNB | DNB | 未提供 | Q1265181 |
| 461 | NYSE:DNI | DNI | 未提供 | — |
| 462 | NYSE:DNP | DNP | 未提供 | — |
| 463 | NYSE:DO | DO | 未提供 | — |
| 464 | NYSE:DOV | DOV | Dover Corporation | Q16974527 |
| 465 | NYSE:DPG | DPG | 未提供 | — |
| 466 | NYSE:DPS | DPS | 未提供 | Q1254149 |
| 467 | NYSE:DPZ | DPZ | Domino's Pizza | Q839466 |
| 468 | NYSE:DRE | DRE | Duke Realty Corp | — |
| 469 | NYSE:DRH | DRH | 未提供 | — |
| 470 | NYSE:DRI | DRI | Darden Restaurants | — |
| 471 | NYSE:DRQ | DRQ | 未提供 | — |
| 472 | NYSE:DS-B | DS-B | 未提供 | — |
| 473 | NYSE:DSM | DSM | 未提供 | — |
| 474 | NYSE:DST | DST | 未提供 | Q5206198 |
| 475 | NYSE:DSU | DSU | 未提供 | — |
| 476 | NYSE:DSW | DSW | 未提供 | Q5206207 |
| 477 | NYSE:DTE | DTE | DTE Energy | Q5206232 |
| 478 | NYSE:DTF | DTF | 未提供 | — |
| 479 | NYSE:DTK | DTK | 未提供 | — |
| 480 | NYSE:DUC | DUC | 未提供 | — |
| 481 | NYSE:DUK | DUK | Duke Energy | — |
| 482 | NYSE:DVA | DVA | DaVita | Q5207184 |
| 483 | NYSE:DVN | DVN | Devon Energy | Q1206980 |
| 484 | NYSE:DX | DX | 未提供 | — |
| 485 | NYSE:DX-A | DX-A | 未提供 | — |
| 486 | NYSE:DXB | DXB | 未提供 | — |
| 487 | NYSE:DY | DY | 未提供 | — |
| 488 | NYSE:DYN | DYN | 未提供 | — |
| 489 | NYSE:E | E | 未提供 | Q565594 |
| 490 | NYSE:EAT | EAT | 未提供 | — |
| 491 | NYSE:EBF | EBF | 未提供 | — |
| 492 | NYSE:EBS | EBS | 未提供 | — |
| 493 | NYSE:EC | EC | 未提供 | — |
| 494 | NYSE:ECL | ECL | Ecolab | Q1282031 |
| 495 | NYSE:ED | ED | Consolidated Edison | — |
| 496 | NYSE:EDD | EDD | 未提供 | — |
| 497 | NYSE:EDF | EDF | 未提供 | — |
| 498 | NYSE:EDI | EDI | 未提供 | — |
| 499 | NYSE:EDR | EDR | 未提供 | — |
| 500 | NYSE:EDU | EDU | 未提供 | Q13631921 |
| 501 | NYSE:EE | EE | 未提供 | — |
| 502 | NYSE:EEA | EEA | 未提供 | — |
| 503 | NYSE:EEP | EEP | 未提供 | — |
| 504 | NYSE:EEQ | EEQ | 未提供 | — |
| 505 | NYSE:EFC | EFC | 未提供 | — |
| 506 | NYSE:EFR | EFR | 未提供 | — |
| 507 | NYSE:EFT | EFT | 未提供 | — |
| 508 | NYSE:EFX | EFX | Equifax | — |
| 509 | NYSE:EGF | EGF | 未提供 | — |
| 510 | NYSE:EGHT | EGHT | 未提供 | — |
| 511 | NYSE:EGN | EGN | 未提供 | Q5376825 |
| 512 | NYSE:EGP | EGP | 未提供 | — |
| 513 | NYSE:EHI | EHI | 未提供 | — |
| 514 | NYSE:EIG | EIG | 未提供 | — |
| 515 | NYSE:EIX | EIX | Edison International | Q662095 |
| 516 | NYSE:EL | EL | Estée Lauder Companies | — |
| 517 | NYSE:ELLI | ELLI | 未提供 | — |
| 518 | NYSE:ELS | ELS | 未提供 | — |
| 519 | NYSE:ELY | ELY | 未提供 | Q1027181 |
| 520 | NYSE:EMD | EMD | 未提供 | — |
| 521 | NYSE:EME | EME | 未提供 | Q5370289 |
| 522 | NYSE:EMF | EMF | 未提供 | — |
| 523 | NYSE:EMN | EMN | Eastman Chemical | Q899022 |
| 524 | NYSE:EMO | EMO | 未提供 | — |
| 525 | NYSE:EMR | EMR | Emerson Electric Company | Q1335585 |
| 526 | NYSE:ENB | ENB | 未提供 | — |
| 527 | NYSE:ENIA | ENIA | 未提供 | — |
| 528 | NYSE:ENLK | ENLK | 未提供 | — |
| 529 | NYSE:ENS | ENS | 未提供 | — |
| 530 | NYSE:ENV | ENV | 未提供 | — |
| 531 | NYSE:EOCC | EOCC | 未提供 | Q3053992 |
| 532 | NYSE:EOD | EOD | 未提供 | — |
| 533 | NYSE:EOG | EOG | EOG Resources | — |
| 534 | NYSE:EOI | EOI | 未提供 | — |
| 535 | NYSE:EOS | EOS | 未提供 | — |
| 536 | NYSE:EOT | EOT | 未提供 | — |
| 537 | NYSE:EP-C | EP-C | 未提供 | — |
| 538 | NYSE:EPAM | EPAM | 未提供 | — |
| 539 | NYSE:EPC | EPC | 未提供 | — |
| 540 | NYSE:EPD | EPD | 未提供 | — |
| 541 | NYSE:EPR | EPR | 未提供 | — |
| 542 | NYSE:EPR-C | EPR-C | 未提供 | — |
| 543 | NYSE:EPR-E | EPR-E | 未提供 | — |
| 544 | NYSE:EQC | EQC | 未提供 | — |
| 545 | NYSE:EQC-D | EQC-D | 未提供 | — |
| 546 | NYSE:EQM | EQM | 未提供 | — |
| 547 | NYSE:EQR | EQR | Equity Residential | Q187740 |
| 548 | NYSE:EQT | EQT | 未提供 | Q5323987 |
| 549 | NYSE:ERJ | ERJ | 未提供 | Q105557 |
| 550 | NYSE:ES | ES | Eversource Energy | Q3344199 |
| 551 | NYSE:ESE | ESE | 未提供 | — |
| 552 | NYSE:ESL | ESL | 未提供 | Q18148411 |
| 553 | NYSE:ESS | ESS | Essex Property Trust | — |
| 554 | NYSE:ESTE | ESTE | 未提供 | — |
| 555 | NYSE:ETB | ETB | 未提供 | — |
| 556 | NYSE:ETG | ETG | 未提供 | — |
| 557 | NYSE:ETH | ETH | 未提供 | — |
| 558 | NYSE:ETJ | ETJ | 未提供 | — |
| 559 | NYSE:ETM | ETM | 未提供 | — |
| 560 | NYSE:ETN | ETN | Eaton Corporation | Q48872 |
| 561 | NYSE:ETO | ETO | 未提供 | — |
| 562 | NYSE:ETP | ETP | 未提供 | Q5377000 |
| 563 | NYSE:ETR | ETR | Entergy | Q1344587 |
| 564 | NYSE:ETV | ETV | 未提供 | — |
| 565 | NYSE:ETW | ETW | 未提供 | — |
| 566 | NYSE:ETY | ETY | 未提供 | — |
| 567 | NYSE:EV | EV | 未提供 | Q5331380 |
| 568 | NYSE:EVF | EVF | 未提供 | — |
| 569 | NYSE:EVG | EVG | 未提供 | — |
| 570 | NYSE:EVN | EVN | 未提供 | — |
| 571 | NYSE:EVR | EVR | 未提供 | — |
| 572 | NYSE:EVT | EVT | 未提供 | — |
| 573 | NYSE:EW | EW | Edwards Lifesciences | — |
| 574 | NYSE:EXC | EXC | Exelon | Q1383669 |
| 575 | NYSE:EXD | EXD | 未提供 | — |
| 576 | NYSE:EXG | EXG | 未提供 | — |
| 577 | NYSE:EXP | EXP | 未提供 | — |
| 578 | NYSE:EXPR | EXPR | 未提供 | — |
| 579 | NYSE:EXR | EXR | Extra Space Storage | Q5422162 |
| 580 | NYSE:F | F | Ford | Q44294 |
| 581 | NYSE:FAF | FAF | 未提供 | — |
| 582 | NYSE:FAM | FAM | 未提供 | — |
| 583 | NYSE:FBC | FBC | 未提供 | — |
| 584 | NYSE:FBHS | FBHS | Fortune Brands Home & Security | — |
| 585 | NYSE:FBR | FBR | 未提供 | — |
| 586 | NYSE:FC | FC | 未提供 | — |
| 587 | NYSE:FCF | FCF | 未提供 | Q5452773 |
| 588 | NYSE:FCFS | FCFS | 未提供 | — |
| 589 | NYSE:FCN | FCN | 未提供 | — |
| 590 | NYSE:FCT | FCT | 未提供 | — |
| 591 | NYSE:FDP | FDP | 未提供 | — |
| 592 | NYSE:FDS | FDS | 未提供 | — |
| 593 | NYSE:FDX | FDX | FedEx | Q459477 |
| 594 | NYSE:FE | FE | FirstEnergy | Q1419248 |
| 595 | NYSE:FEI | FEI | 未提供 | — |
| 596 | NYSE:FEO | FEO | 未提供 | — |
| 597 | NYSE:FET | FET | 未提供 | Q20708639 |
| 598 | NYSE:FF | FF | 未提供 | — |
| 599 | NYSE:FFA | FFA | 未提供 | — |
| 600 | NYSE:FFC | FFC | 未提供 | — |
| 601 | NYSE:FFG | FFG | 未提供 | Q5424650 |
| 602 | NYSE:FHN | FHN | 未提供 | Q1419446 |
| 603 | NYSE:FHY | FHY | 未提供 | — |
| 604 | NYSE:FICO | FICO | 未提供 | — |
| 605 | NYSE:FIF | FIF | 未提供 | — |
| 606 | NYSE:FII | FII | 未提供 | — |
| 607 | NYSE:FIS | FIS | Fidelity National Information Services | — |
| 608 | NYSE:FIX | FIX | 未提供 | — |
| 609 | NYSE:FL | FL | 未提供 | Q63335 |
| 610 | NYSE:FLC | FLC | 未提供 | — |
| 611 | NYSE:FLO | FLO | 未提供 | — |
| 612 | NYSE:FLR | FLR | 未提供 | — |
| 613 | NYSE:FLS | FLS | 未提供 | — |
| 614 | NYSE:FLT | FLT | Fleetcor | — |
| 615 | NYSE:FLY | FLY | 未提供 | Q5426582 |
| 616 | NYSE:FMC | FMC | FMC Corporation | Q899183 |
| 617 | NYSE:FMN | FMN | 未提供 | — |
| 618 | NYSE:FMO | FMO | 未提供 | — |
| 619 | NYSE:FMS | FMS | 未提供 | — |
| 620 | NYSE:FMX | FMX | 未提供 | Q1435628 |
| 621 | NYSE:FMY | FMY | 未提供 | — |
| 622 | NYSE:FN | FN | 未提供 | — |
| 623 | NYSE:FNB | FNB | 未提供 | Q5426765 |
| 624 | NYSE:FNV | FNV | 未提供 | Q5484477 |
| 625 | NYSE:FOF | FOF | 未提供 | — |
| 626 | NYSE:FPT | FPT | 未提供 | — |
| 627 | NYSE:FR | FR | 未提供 | — |
| 628 | NYSE:FRA | FRA | 未提供 | — |
| 629 | NYSE:FRC | FRC | First Republic Bank | — |
| 630 | NYSE:FRT | FRT | Federal Realty Investment Trust | — |
| 631 | NYSE:FSD | FSD | 未提供 | — |
| 632 | NYSE:FSS | FSS | 未提供 | Q1400073 |
| 633 | NYSE:FT | FT | 未提供 | — |
| 634 | NYSE:FUL | FUL | 未提供 | Q20486915 |
| 635 | NYSE:FUN | FUN | 未提供 | Q657648 |
| 636 | NYSE:G | G | 未提供 | — |
| 637 | NYSE:GAB-G | GAB-G | 未提供 | — |
| 638 | NYSE:GAB-H | GAB-H | 未提供 | — |
| 639 | NYSE:GAM | GAM | 未提供 | — |
| 640 | NYSE:GAM-B | GAM-B | 未提供 | — |
| 641 | NYSE:GATX | GATX | 未提供 | — |
| 642 | NYSE:GBAB | GBAB | 未提供 | — |
| 643 | NYSE:GBL | GBL | 未提供 | — |
| 644 | NYSE:GBX | GBX | 未提供 | Q7738224 |
| 645 | NYSE:GCH | GCH | 未提供 | — |
| 646 | NYSE:GCO | GCO | 未提供 | — |
| 647 | NYSE:GD | GD | General Dynamics | Q502940 |
| 648 | NYSE:GDL | GDL | 未提供 | — |
| 649 | NYSE:GDO | GDO | 未提供 | — |
| 650 | NYSE:GDOT | GDOT | 未提供 | Q16975278 |
| 651 | NYSE:GDV | GDV | 未提供 | — |
| 652 | NYSE:GDV-A | GDV-A | 未提供 | — |
| 653 | NYSE:GDV-D | GDV-D | 未提供 | — |
| 654 | NYSE:GE | GE | General Electric | Q54173 |
| 655 | NYSE:GEF | GEF | 未提供 | — |
| 656 | NYSE:GEF.B | GEF.B | 未提供 | — |
| 657 | NYSE:GEL | GEL | 未提供 | — |
| 658 | NYSE:GEO | GEO | 未提供 | Q1502794 |
| 659 | NYSE:GES | GES | 未提供 | — |
| 660 | NYSE:GF | GF | 未提供 | — |
| 661 | NYSE:GFF | GFF | 未提供 | Q16840288 |
| 662 | NYSE:GFY | GFY | 未提供 | — |
| 663 | NYSE:GG | GG | 未提供 | — |
| 664 | NYSE:GGG | GGG | 未提供 | Q5591659 |
| 665 | NYSE:GGP | GGP | 未提供 | Q3100500 |
| 666 | NYSE:GGT | GGT | 未提供 | — |
| 667 | NYSE:GHC | GHC | 未提供 | Q3121014 |
| 668 | NYSE:GHL | GHL | 未提供 | — |
| 669 | NYSE:GHM | GHM | 未提供 | — |
| 670 | NYSE:GHY | GHY | 未提供 | — |
| 671 | NYSE:GIB | GIB | 未提供 | Q1798370 |
| 672 | NYSE:GIL | GIL | 未提供 | — |
| 673 | NYSE:GIM | GIM | 未提供 | — |
| 674 | NYSE:GIS | GIS | General Mills | Q629998 |
| 675 | NYSE:GLOG | GLOG | 未提供 | — |
| 676 | NYSE:GLP | GLP | 未提供 | — |
| 677 | NYSE:GLT | GLT | 未提供 | Q726654 |
| 678 | NYSE:GLW | GLW | Corning | — |
| 679 | NYSE:GM | GM | General Motors | Q81965 |
| 680 | NYSE:GME | GME | 未提供 | Q202210 |
| 681 | NYSE:GMED | GMED | 未提供 | Q18344949 |
| 682 | NYSE:GNC | GNC | 未提供 | — |
| 683 | NYSE:GNE-A | GNE-A | 未提供 | — |
| 684 | NYSE:GNRC | GNRC | Generac Holdings | — |
| 685 | NYSE:GOF | GOF | 未提供 | — |
| 686 | NYSE:GPC | GPC | Genuine Parts | Q5533818 |
| 687 | NYSE:GPI | GPI | 未提供 | Q5611081 |
| 688 | NYSE:GPK | GPK | 未提供 | — |
| 689 | NYSE:GPM | GPM | 未提供 | — |
| 690 | NYSE:GPN | GPN | Global Payments | Q5570544 |
| 691 | NYSE:GPS | GPS | Gap | Q420822 |
| 692 | NYSE:GPX | GPX | 未提供 | — |
| 693 | NYSE:GRA | GRA | 未提供 | Q498746 |
| 694 | NYSE:GRC | GRC | 未提供 | — |
| 695 | NYSE:GRR | GRR | 未提供 | — |
| 696 | NYSE:GRX | GRX | 未提供 | — |
| 697 | NYSE:GS | GS | Goldman Sachs | Q193326 |
| 698 | NYSE:GS-A | GS-A | 未提供 | — |
| 699 | NYSE:GS-B | GS-B | 未提供 | — |
| 700 | NYSE:GS-C | GS-C | 未提供 | — |
| 701 | NYSE:GS-D | GS-D | 未提供 | — |
| 702 | NYSE:GSH | GSH | 未提供 | Q837982 |
| 703 | NYSE:GSK | GSK | 未提供 | Q212322 |
| 704 | NYSE:GTS | GTS | 未提供 | Q7843323 |
| 705 | NYSE:GTY | GTY | 未提供 | — |
| 706 | NYSE:GUT | GUT | 未提供 | — |
| 707 | NYSE:GVA | GVA | 未提供 | Q5595849 |
| 708 | NYSE:GWR | GWR | 未提供 | Q1230295 |
| 709 | NYSE:GWRE | GWRE | 未提供 | — |
| 710 | NYSE:GWW | GWW | W. W. Grainger | Q1627894 |
| 711 | NYSE:GXP | GXP | 未提供 | Q5599779 |
| 712 | NYSE:GYB | GYB | 未提供 | — |
| 713 | NYSE:GYC | GYC | 未提供 | — |
| 714 | NYSE:GZT | GZT | 未提供 | — |
| 715 | NYSE:H | H | 未提供 | Q1425063 |
| 716 | NYSE:HAE | HAE | 未提供 | Q16980281 |
| 717 | NYSE:HAL | HAL | Halliburton | Q656165 |
| 718 | NYSE:HBI | HBI | Hanesbrands | — |
| 719 | NYSE:HCA | HCA | HCA Healthcare | — |
| 720 | NYSE:HCI | HCI | 未提供 | — |
| 721 | NYSE:HCN | HCN | 未提供 | — |
| 722 | NYSE:HCN-I | HCN-I | 未提供 | — |
| 723 | NYSE:HCP | HCP | 未提供 | — |
| 724 | NYSE:HD | HD | Home Depot | Q864407 |
| 725 | NYSE:HDB | HDB | 未提供 | — |
| 726 | NYSE:HE | HE | 未提供 | Q5684524 |
| 727 | NYSE:HEI | HEI | 未提供 | — |
| 728 | NYSE:HEI.A | HEI.A | 未提供 | — |
| 729 | NYSE:HEP | HEP | 未提供 | — |
| 730 | NYSE:HEQ | HEQ | 未提供 | — |
| 731 | NYSE:HES | HES | Hess Corporation | Q1615684 |
| 732 | NYSE:HF | HF | 未提供 | Q5881333 |
| 733 | NYSE:HFC | HFC | 未提供 | Q5881864 |
| 734 | NYSE:HHC | HHC | 未提供 | — |
| 735 | NYSE:HI | HI | 未提供 | Q5763038 |
| 736 | NYSE:HIG | HIG | The Hartford | — |
| 737 | NYSE:HII | HII | Huntington Ingalls Industries | Q7257522 |
| 738 | NYSE:HIW | HIW | 未提供 | — |
| 739 | NYSE:HIX | HIX | 未提供 | — |
| 740 | NYSE:HJV | HJV | 未提供 | — |
| 741 | NYSE:HLF | HLF | 未提供 | Q572948 |
| 742 | NYSE:HLS | HLS | 未提供 | Q5690400 |
| 743 | NYSE:HMC | HMC | 未提供 | Q9584 |
| 744 | NYSE:HMN | HMN | 未提供 | — |
| 745 | NYSE:HNI | HNI | 未提供 | — |
| 746 | NYSE:HNP | HNP | 未提供 | Q1507642 |
| 747 | NYSE:HOG | HOG | 未提供 | Q192814 |
| 748 | NYSE:HON | HON | Honeywell | Q898208 |
| 749 | NYSE:HP | HP | 未提供 | — |
| 750 | NYSE:HPF | HPF | 未提供 | — |
| 751 | NYSE:HPI | HPI | 未提供 | — |
| 752 | NYSE:HPP | HPP | 未提供 | — |
| 753 | NYSE:HPS | HPS | 未提供 | — |
| 754 | NYSE:HQH | HQH | 未提供 | — |
| 755 | NYSE:HQL | HQL | 未提供 | — |
| 756 | NYSE:HR | HR | 未提供 | — |
| 757 | NYSE:HRB | HRB | 未提供 | Q5627799 |
| 758 | NYSE:HRC | HRC | 未提供 | Q5761967 |
| 759 | NYSE:HRG | HRG | 未提供 | Q147224 |
| 760 | NYSE:HRL | HRL | Hormel | — |
| 761 | NYSE:HRS | HRS | 未提供 | — |
| 762 | NYSE:HSBC | HSBC | 未提供 | Q190464 |
| 763 | NYSE:HSEA | HSEA | 未提供 | — |
| 764 | NYSE:HSEB | HSEB | 未提供 | — |
| 765 | NYSE:HST | HST | Host Hotels & Resorts | Q1630559 |
| 766 | NYSE:HSY | HSY | The Hershey Company | Q1056637 |
| 767 | NYSE:HT | HT | 未提供 | — |
| 768 | NYSE:HTD | HTD | 未提供 | — |
| 769 | NYSE:HTGC | HTGC | 未提供 | — |
| 770 | NYSE:HTH | HTH | 未提供 | — |
| 771 | NYSE:HTY | HTY | 未提供 | — |
| 772 | NYSE:HUBB | HUBB | 未提供 | — |
| 773 | NYSE:HUM | HUM | Humana | Q910866 |
| 774 | NYSE:HUN | HUN | 未提供 | Q851361 |
| 775 | NYSE:HVT | HVT | 未提供 | Q5683881 |
| 776 | NYSE:HVT.A | HVT.A | 未提供 | — |
| 777 | NYSE:HXL | HXL | 未提供 | Q5748967 |
| 778 | NYSE:HY | HY | 未提供 | Q19876423 |
| 779 | NYSE:HYB | HYB | 未提供 | — |
| 780 | NYSE:HYI | HYI | 未提供 | — |
| 781 | NYSE:HYT | HYT | 未提供 | — |
| 782 | NYSE:HZO | HZO | 未提供 | — |
| 783 | NYSE:IAE | IAE | 未提供 | — |
| 784 | NYSE:IBA | IBA | 未提供 | Q4838903 |
| 785 | NYSE:IBM | IBM | IBM | Q37156 |
| 786 | NYSE:IBN | IBN | 未提供 | Q1653258 |
| 787 | NYSE:ICB | ICB | 未提供 | — |
| 788 | NYSE:IDA | IDA | 未提供 | — |
| 789 | NYSE:IDE | IDE | 未提供 | — |
| 790 | NYSE:IDT | IDT | 未提供 | — |
| 791 | NYSE:IEX | IEX | IDEX Corporation | Q5970002 |
| 792 | NYSE:IFF | IFF | International Flavors & Fragrances | — |
| 793 | NYSE:IFN | IFN | 未提供 | — |
| 794 | NYSE:IGA | IGA | 未提供 | — |
| 795 | NYSE:IGD | IGD | 未提供 | — |
| 796 | NYSE:IGI | IGI | 未提供 | — |
| 797 | NYSE:IGR | IGR | 未提供 | — |
| 798 | NYSE:IHC | IHC | 未提供 | — |
| 799 | NYSE:IHD | IHD | 未提供 | — |
| 800 | NYSE:IHG | IHG | 未提供 | Q1424962 |
| 801 | NYSE:IID | IID | 未提供 | — |
| 802 | NYSE:IIF | IIF | 未提供 | — |
| 803 | NYSE:IMAX | IMAX | 未提供 | — |
| 804 | NYSE:INB | INB | 未提供 | — |
| 805 | NYSE:INF | INF | 未提供 | — |
| 806 | NYSE:INFY | INFY | 未提供 | Q26989 |
| 807 | NYSE:ING | ING | 未提供 | Q645708 |
| 808 | NYSE:INGR | INGR | 未提供 | — |
| 809 | NYSE:INN | INN | 未提供 | — |
| 810 | NYSE:INSI | INSI | 未提供 | — |
| 811 | NYSE:INT | INT | 未提供 | Q8035757 |
| 812 | NYSE:INXN | INXN | 未提供 | — |
| 813 | NYSE:IP | IP | International Paper | Q202972 |
| 814 | NYSE:IPG | IPG | Interpublic Group | — |
| 815 | NYSE:IPHI | IPHI | 未提供 | — |
| 816 | NYSE:IR | IR | Ingersoll Rand | Q1663044 |
| 817 | NYSE:IRET | IRET | 未提供 | — |
| 818 | NYSE:IRL | IRL | 未提供 | — |
| 819 | NYSE:IRS | IRS | 未提供 | — |
| 820 | NYSE:ISD | ISD | 未提供 | — |
| 821 | NYSE:ISF | ISF | 未提供 | — |
| 822 | NYSE:ISG | ISG | 未提供 | — |
| 823 | NYSE:IT | IT | Gartner | — |
| 824 | NYSE:ITCB | ITCB | 未提供 | — |
| 825 | NYSE:ITG | ITG | 未提供 | Q6060808 |
| 826 | NYSE:ITUB | ITUB | 未提供 | Q1424293 |
| 827 | NYSE:ITW | ITW | Illinois Tool Works | — |
| 828 | NYSE:IVC | IVC | 未提供 | Q4994426 |
| 829 | NYSE:IVR | IVR | 未提供 | — |
| 830 | NYSE:IVR-A | IVR-A | 未提供 | — |
| 831 | NYSE:IVZ | IVZ | Invesco | — |
| 832 | NYSE:IX | IX | 未提供 | Q906055 |
| 833 | NYSE:JBL | JBL | 未提供 | Q2171646 |
| 834 | NYSE:JBN | JBN | 未提供 | — |
| 835 | NYSE:JBR | JBR | 未提供 | — |
| 836 | NYSE:JBT | JBT | 未提供 | Q3156864 |
| 837 | NYSE:JCE | JCE | 未提供 | — |
| 838 | NYSE:JDD | JDD | 未提供 | — |
| 839 | NYSE:JEQ | JEQ | 未提供 | — |
| 840 | NYSE:JFR | JFR | 未提供 | — |
| 841 | NYSE:JHI | JHI | 未提供 | — |
| 842 | NYSE:JHS | JHS | 未提供 | — |
| 843 | NYSE:JHX | JHX | 未提供 | — |
| 844 | NYSE:JLL | JLL | 未提供 | Q1703389 |
| 845 | NYSE:JLS | JLS | 未提供 | — |
| 846 | NYSE:JMF | JMF | 未提供 | — |
| 847 | NYSE:JMM | JMM | 未提供 | — |
| 848 | NYSE:JMT | JMT | 未提供 | — |
| 849 | NYSE:JNJ | JNJ | Johnson & Johnson | Q333718 |
| 850 | NYSE:JNPR | JNPR | Juniper Networks | Q1049080 |
| 851 | NYSE:JOE | JOE | 未提供 | — |
| 852 | NYSE:JOF | JOF | 未提供 | — |
| 853 | NYSE:JPC | JPC | 未提供 | — |
| 854 | NYSE:JPI | JPI | 未提供 | — |
| 855 | NYSE:JPM | JPM | JPMorgan Chase | Q192314 |
| 856 | NYSE:JPS | JPS | 未提供 | — |
| 857 | NYSE:JQC | JQC | 未提供 | — |
| 858 | NYSE:JRI | JRI | 未提供 | — |
| 859 | NYSE:JRO | JRO | 未提供 | — |
| 860 | NYSE:JRS | JRS | 未提供 | — |
| 861 | NYSE:JSD | JSD | 未提供 | — |
| 862 | NYSE:JTA | JTA | 未提供 | — |
| 863 | NYSE:JTD | JTD | 未提供 | — |
| 864 | NYSE:JW.A | JW.A | 未提供 | — |
| 865 | NYSE:JW.B | JW.B | 未提供 | — |
| 866 | NYSE:JWN | JWN | 未提供 | Q174310 |
| 867 | NYSE:K | K | Kellogg's | Q856897 |
| 868 | NYSE:KAI | KAI | 未提供 | — |
| 869 | NYSE:KAMN | KAMN | 未提供 | — |
| 870 | NYSE:KAR | KAR | 未提供 | — |
| 871 | NYSE:KB | KB | 未提供 | Q496606 |
| 872 | NYSE:KBH | KBH | 未提供 | — |
| 873 | NYSE:KBR | KBR | 未提供 | Q596499 |
| 874 | NYSE:KED | KED | 未提供 | — |
| 875 | NYSE:KEP | KEP | 未提供 | Q486737 |
| 876 | NYSE:KEX | KEX | 未提供 | — |
| 877 | NYSE:KEY | KEY | KeyCorp | Q1740314 |
| 878 | NYSE:KF | KF | 未提供 | — |
| 879 | NYSE:KFY | KFY | 未提供 | Q1728446 |
| 880 | NYSE:KIM | KIM | Kimco Realty | — |
| 881 | NYSE:KIM-I | KIM-I | 未提供 | — |
| 882 | NYSE:KIM-J | KIM-J | 未提供 | — |
| 883 | NYSE:KIM-K | KIM-K | 未提供 | — |
| 884 | NYSE:KKR | KKR | 未提供 | — |
| 885 | NYSE:KMB | KMB | Kimberly-Clark | Q1741634 |
| 886 | NYSE:KMF | KMF | 未提供 | — |
| 887 | NYSE:KMG | KMG | 未提供 | — |
| 888 | NYSE:KMI | KMI | Kinder Morgan | — |
| 889 | NYSE:KMM | KMM | 未提供 | — |
| 890 | NYSE:KMPR | KMPR | 未提供 | Q6387063 |
| 891 | NYSE:KMT | KMT | 未提供 | — |
| 892 | NYSE:KMX | KMX | CarMax | — |
| 893 | NYSE:KND | KND | 未提供 | Q921363 |
| 894 | NYSE:KNL | KNL | 未提供 | — |
| 895 | NYSE:KO | KO | Coca-Cola Company | Q3295867 |
| 896 | NYSE:KOF | KOF | 未提供 | — |
| 897 | NYSE:KOP | KOP | 未提供 | — |
| 898 | NYSE:KORS | KORS | 未提供 | — |
| 899 | NYSE:KR | KR | Kroger | Q153417 |
| 900 | NYSE:KRA | KRA | 未提供 | Q2417728 |
| 901 | NYSE:KRC | KRC | 未提供 | — |
| 902 | NYSE:KS | KS | 未提供 | — |
| 903 | NYSE:KSM | KSM | 未提供 | — |
| 904 | NYSE:KSS | KSS | 未提供 | — |
| 905 | NYSE:KST | KST | 未提供 | — |
| 906 | NYSE:KSU | KSU | Kansas City Southern | — |
| 907 | NYSE:KT | KT | 未提供 | Q485908 |
| 908 | NYSE:KTF | KTF | 未提供 | — |
| 909 | NYSE:KTP | KTP | 未提供 | — |
| 910 | NYSE:KW | KW | 未提供 | — |
| 911 | NYSE:KWR | KWR | 未提供 | — |
| 912 | NYSE:KYE | KYE | 未提供 | — |
| 913 | NYSE:KYN | KYN | 未提供 | — |
| 914 | NYSE:KYO | KYO | 未提供 | Q745099 |
| 915 | NYSE:L | L | Loews Corporation | — |
| 916 | NYSE:LAD | LAD | 未提供 | Q6647881 |
| 917 | NYSE:LAZ | LAZ | 未提供 | — |
| 918 | NYSE:LB | LB | 未提供 | Q931354 |
| 919 | NYSE:LCI | LCI | 未提供 | — |
| 920 | NYSE:LCII | LCII | 未提供 | — |
| 921 | NYSE:LCM | LCM | 未提供 | — |
| 922 | NYSE:LDF | LDF | 未提供 | — |
| 923 | NYSE:LDL | LDL | 未提供 | — |
| 924 | NYSE:LDP | LDP | 未提供 | — |
| 925 | NYSE:LEA | LEA | 未提供 | — |
| 926 | NYSE:LEG | LEG | Leggett & Platt | — |
| 927 | NYSE:LEN | LEN | Lennar | Q16981175 |
| 928 | NYSE:LEN.B | LEN.B | 未提供 | — |
| 929 | NYSE:LEO | LEO | 未提供 | — |
| 930 | NYSE:LFC | LFC | 未提供 | Q1073306 |
| 931 | NYSE:LGI | LGI | 未提供 | — |
| 932 | NYSE:LH | LH | LabCorp | Q6466630 |
| 933 | NYSE:LHO | LHO | 未提供 | — |
| 934 | NYSE:LII | LII | 未提供 | — |
| 935 | NYSE:LL | LL | 未提供 | Q6703090 |
| 936 | NYSE:LLL | LLL | 未提供 | Q1781428 |
| 937 | NYSE:LLY | LLY | Eli Lilly & Co | Q632240 |
| 938 | NYSE:LM | LM | 未提供 | — |
| 939 | NYSE:LMT | LMT | Lockheed Martin | Q7240 |
| 940 | NYSE:LNC | LNC | Lincoln National | Q1825763 |
| 941 | NYSE:LNN | LNN | 未提供 | Q6552749 |
| 942 | NYSE:LNT | LNT | Alliant Energy | — |
| 943 | NYSE:LOR | LOR | 未提供 | — |
| 944 | NYSE:LOW | LOW | Lowe's | Q1373493 |
| 945 | NYSE:LPL | LPL | 未提供 | Q483006 |
| 946 | NYSE:LPT | LPT | 未提供 | — |
| 947 | NYSE:LPX | LPX | 未提供 | — |
| 948 | NYSE:LRN | LRN | 未提供 | Q6324302 |
| 949 | NYSE:LSI | LSI | 未提供 | — |
| 950 | NYSE:LTC | LTC | 未提供 | — |
| 951 | NYSE:LUK | LUK | 未提供 | Q6533918 |
| 952 | NYSE:LUV | LUV | Southwest Airlines | Q503308 |
| 953 | NYSE:LVS | LVS | Las Vegas Sands | Q1506045 |
| 954 | NYSE:LXFR | LXFR | 未提供 | — |
| 955 | NYSE:LXP | LXP | 未提供 | — |
| 956 | NYSE:LXP-C | LXP-C | 未提供 | — |
| 957 | NYSE:LYB | LYB | LyondellBasell | Q899255 |
| 958 | NYSE:LYV | LYV | Live Nation Entertainment | Q1061903 |
| 959 | NYSE:LZB | LZB | 未提供 | Q6391583 |
| 960 | NYSE:M | M | 未提供 | Q5479360 |
| 961 | NYSE:MA | MA | Mastercard | Q489921 |
| 962 | NYSE:MAA | MAA | Mid-America Apartments | — |
| 963 | NYSE:MAC | MAC | 未提供 | — |
| 964 | NYSE:MAIN | MAIN | 未提供 | — |
| 965 | NYSE:MAN | MAN | 未提供 | — |
| 966 | NYSE:MANU | MANU | 未提供 | — |
| 967 | NYSE:MAS | MAS | Masco | Q6782872 |
| 968 | NYSE:MATX | MATX | 未提供 | — |
| 969 | NYSE:MAV | MAV | 未提供 | — |
| 970 | NYSE:MBI | MBI | 未提供 | Q1371852 |
| 971 | NYSE:MBT | MBT | 未提供 | Q1368919 |
| 972 | NYSE:MCA | MCA | 未提供 | — |
| 973 | NYSE:MCC | MCC | 未提供 | — |
| 974 | NYSE:MCD | MCD | McDonald's | Q38076 |
| 975 | NYSE:MCI | MCI | 未提供 | — |
| 976 | NYSE:MCK | MCK | McKesson Corporation | — |
| 977 | NYSE:MCN | MCN | 未提供 | — |
| 978 | NYSE:MCO | MCO | Moody's Corporation | Q675585 |
| 979 | NYSE:MCR | MCR | 未提供 | — |
| 980 | NYSE:MCS | MCS | 未提供 | Q6758107 |
| 981 | NYSE:MCY | MCY | 未提供 | Q16981897 |
| 982 | NYSE:MD | MD | 未提供 | — |
| 983 | NYSE:MDC | MDC | 未提供 | — |
| 984 | NYSE:MDP | MDP | 未提供 | Q1921579 |
| 985 | NYSE:MDU | MDU | 未提供 | — |
| 986 | NYSE:MED | MED | 未提供 | Q6806861 |
| 987 | NYSE:MEI | MEI | 未提供 | — |
| 988 | NYSE:MEN | MEN | 未提供 | — |
| 989 | NYSE:MER-K | MER-K | 未提供 | — |
| 990 | NYSE:MER-P | MER-P | 未提供 | — |
| 991 | NYSE:MET | MET | MetLife | Q607131 |
| 992 | NYSE:MET-A | MET-A | 未提供 | — |
| 993 | NYSE:MFA | MFA | 未提供 | — |
| 994 | NYSE:MFC | MFC | 未提供 | Q1652574 |
| 995 | NYSE:MFD | MFD | 未提供 | — |
| 996 | NYSE:MFL | MFL | 未提供 | — |
| 997 | NYSE:MFM | MFM | 未提供 | — |
| 998 | NYSE:MFT | MFT | 未提供 | — |
| 999 | NYSE:MG | MG | 未提供 | — |
| 1000 | NYSE:MGA | MGA | 未提供 | Q697311 |
| 1001 | NYSE:MGM | MGM | MGM Resorts International | Q568532 |
| 1002 | NYSE:MGU | MGU | 未提供 | — |
| 1003 | NYSE:MH-A | MH-A | 未提供 | — |
| 1004 | NYSE:MHD | MHD | 未提供 | — |
| 1005 | NYSE:MHF | MHF | 未提供 | — |
| 1006 | NYSE:MHI | MHI | 未提供 | — |
| 1007 | NYSE:MHK | MHK | Mohawk Industries | — |
| 1008 | NYSE:MHN | MHN | 未提供 | — |
| 1009 | NYSE:MHO | MHO | 未提供 | — |
| 1010 | NYSE:MIC | MIC | 未提供 | Q6724984 |
| 1011 | NYSE:MITT | MITT | 未提供 | — |
| 1012 | NYSE:MIY | MIY | 未提供 | — |
| 1013 | NYSE:MKC | MKC | McCormick & Company | — |
| 1014 | NYSE:MKC.V | MKC.V | 未提供 | — |
| 1015 | NYSE:MKL | MKL | 未提供 | Q6770621 |
| 1016 | NYSE:MLI | MLI | 未提供 | — |
| 1017 | NYSE:MLM | MLM | Martin Marietta Materials | Q6776123 |
| 1018 | NYSE:MLR | MLR | 未提供 | — |
| 1019 | NYSE:MMC | MMC | Marsh & McLennan | Q1786345 |
| 1020 | NYSE:MMD | MMD | 未提供 | — |
| 1021 | NYSE:MMM | MMM | 3M | Q159433 |
| 1022 | NYSE:MMP | MMP | 未提供 | — |
| 1023 | NYSE:MMS | MMS | 未提供 | Q6795931 |
| 1024 | NYSE:MMT | MMT | 未提供 | — |
| 1025 | NYSE:MMU | MMU | 未提供 | — |
| 1026 | NYSE:MNE | MNE | 未提供 | — |
| 1027 | NYSE:MNP | MNP | 未提供 | — |
| 1028 | NYSE:MNR | MNR | 未提供 | — |
| 1029 | NYSE:MO | MO | Altria Group | Q445007 |
| 1030 | NYSE:MOD | MOD | 未提供 | — |
| 1031 | NYSE:MOG.A | MOG.A | 未提供 | — |
| 1032 | NYSE:MOG.B | MOG.B | 未提供 | — |
| 1033 | NYSE:MOH | MOH | 未提供 | — |
| 1034 | NYSE:MON | MON | 未提供 | Q207983 |
| 1035 | NYSE:MOS | MOS | The Mosaic Company | — |
| 1036 | NYSE:MOV | MOV | 未提供 | — |
| 1037 | NYSE:MP-D | MP-D | 未提供 | — |
| 1038 | NYSE:MPA | MPA | 未提供 | — |
| 1039 | NYSE:MPC | MPC | Marathon Petroleum | Q458363 |
| 1040 | NYSE:MPLX | MPLX | 未提供 | — |
| 1041 | NYSE:MPV | MPV | 未提供 | — |
| 1042 | NYSE:MPW | MPW | 未提供 | — |
| 1043 | NYSE:MPX | MPX | 未提供 | — |
| 1044 | NYSE:MQT | MQT | 未提供 | — |
| 1045 | NYSE:MQY | MQY | 未提供 | — |
| 1046 | NYSE:MRC | MRC | 未提供 | — |
| 1047 | NYSE:MRK | MRK | Merck & Co. | Q247489 |
| 1048 | NYSE:MRO | MRO | Marathon Oil | Q1577587 |
| 1049 | NYSE:MS | MS | Morgan Stanley | Q334204 |
| 1050 | NYSE:MS-A | MS-A | 未提供 | — |
| 1051 | NYSE:MSCI | MSCI | MSCI | Q768773 |
| 1052 | NYSE:MSD | MSD | 未提供 | — |
| 1053 | NYSE:MSF | MSF | 未提供 | — |
| 1054 | NYSE:MSI | MSI | Motorola Solutions | Q634815 |
| 1055 | NYSE:MSL | MSL | 未提供 | — |
| 1056 | NYSE:MSM | MSM | 未提供 | — |
| 1057 | NYSE:MSP | MSP | 未提供 | — |
| 1058 | NYSE:MTB | MTB | M&T Bank | — |
| 1059 | NYSE:MTD | MTD | Mettler Toledo | — |
| 1060 | NYSE:MTDR | MTDR | 未提供 | — |
| 1061 | NYSE:MTH | MTH | 未提供 | Q6819538 |
| 1062 | NYSE:MTN | MTN | 未提供 | Q16986288 |
| 1063 | NYSE:MTR | MTR | 未提供 | — |
| 1064 | NYSE:MTRN | MTRN | 未提供 | — |
| 1065 | NYSE:MTT | MTT | 未提供 | — |
| 1066 | NYSE:MTX | MTX | 未提供 | — |
| 1067 | NYSE:MTZ | MTZ | 未提供 | Q6782217 |
| 1068 | NYSE:MUA | MUA | 未提供 | — |
| 1069 | NYSE:MUC | MUC | 未提供 | — |
| 1070 | NYSE:MUE | MUE | 未提供 | — |
| 1071 | NYSE:MUH | MUH | 未提供 | — |
| 1072 | NYSE:MUI | MUI | 未提供 | — |
| 1073 | NYSE:MUJ | MUJ | 未提供 | — |
| 1074 | NYSE:MUR | MUR | 未提供 | — |
| 1075 | NYSE:MUS | MUS | 未提供 | — |
| 1076 | NYSE:MVC | MVC | 未提供 | — |
| 1077 | NYSE:MVT | MVT | 未提供 | — |
| 1078 | NYSE:MWA | MWA | 未提供 | — |
| 1079 | NYSE:MXE | MXE | 未提供 | — |
| 1080 | NYSE:MXF | MXF | 未提供 | — |
| 1081 | NYSE:MXL | MXL | 未提供 | — |
| 1082 | NYSE:MYC | MYC | 未提供 | — |
| 1083 | NYSE:MYD | MYD | 未提供 | — |
| 1084 | NYSE:MYE | MYE | 未提供 | — |
| 1085 | NYSE:MYF | MYF | 未提供 | — |
| 1086 | NYSE:MYI | MYI | 未提供 | — |
| 1087 | NYSE:MYJ | MYJ | 未提供 | — |
| 1088 | NYSE:MYN | MYN | 未提供 | — |
| 1089 | NYSE:MZF | MZF | 未提供 | — |
| 1090 | NYSE:NAD | NAD | 未提供 | — |
| 1091 | NYSE:NAN | NAN | 未提供 | — |
| 1092 | NYSE:NAV | NAV | 未提供 | — |
| 1093 | NYSE:NAZ | NAZ | 未提供 | — |
| 1094 | NYSE:NBB | NBB | 未提供 | — |
| 1095 | NYSE:NBD | NBD | 未提供 | — |
| 1096 | NYSE:NBHC | NBHC | 未提供 | — |
| 1097 | NYSE:NBR | NBR | 未提供 | — |
| 1098 | NYSE:NCA | NCA | 未提供 | — |
| 1099 | NYSE:NCB | NCB | 未提供 | — |
| 1100 | NYSE:NCI | NCI | 未提供 | Q6982423 |
| 1101 | NYSE:NCR | NCR | 未提供 | — |
| 1102 | NYSE:NCS | NCS | 未提供 | — |
| 1103 | NYSE:NDP | NDP | 未提供 | — |
| 1104 | NYSE:NEA | NEA | 未提供 | — |
| 1105 | NYSE:NEE | NEE | NextEra Energy | — |
| 1106 | NYSE:NEM | NEM | Newmont | — |
| 1107 | NYSE:NEU | NEU | 未提供 | — |
| 1108 | NYSE:NEV | NEV | 未提供 | — |
| 1109 | NYSE:NFG | NFG | 未提供 | — |
| 1110 | NYSE:NFJ | NFJ | 未提供 | — |
| 1111 | NYSE:NFX | NFX | 未提供 | — |
| 1112 | NYSE:NGG | NGG | 未提供 | Q1465461 |
| 1113 | NYSE:NGL | NGL | 未提供 | — |
| 1114 | NYSE:NGS | NGS | 未提供 | — |
| 1115 | NYSE:NHF | NHF | 未提供 | — |
| 1116 | NYSE:NHI | NHI | 未提供 | — |
| 1117 | NYSE:NID | NID | 未提供 | — |
| 1118 | NYSE:NIE | NIE | 未提供 | — |
| 1119 | NYSE:NIM | NIM | 未提供 | — |
| 1120 | NYSE:NJR | NJR | 未提供 | — |
| 1121 | NYSE:NJV | NJV | 未提供 | — |
| 1122 | NYSE:NKE | NKE | Nike | Q483915 |
| 1123 | NYSE:NKG | NKG | 未提供 | — |
| 1124 | NYSE:NKX | NKX | 未提供 | — |
| 1125 | NYSE:NLY | NLY | 未提供 | — |
| 1126 | NYSE:NLY-C | NLY-C | 未提供 | — |
| 1127 | NYSE:NLY-D | NLY-D | 未提供 | — |
| 1128 | NYSE:NMFC | NMFC | 未提供 | — |
| 1129 | NYSE:NMI | NMI | 未提供 | — |
| 1130 | NYSE:NMT | NMT | 未提供 | — |
| 1131 | NYSE:NMY | NMY | 未提供 | — |
| 1132 | NYSE:NMZ | NMZ | 未提供 | — |
| 1133 | NYSE:NNC | NNC | 未提供 | — |
| 1134 | NYSE:NNI | NNI | 未提供 | Q6990264 |
| 1135 | NYSE:NNN | NNN | 未提供 | — |
| 1136 | NYSE:NNY | NNY | 未提供 | — |
| 1137 | NYSE:NOAH | NOAH | 未提供 | — |
| 1138 | NYSE:NOC | NOC | Northrop Grumman | Q4045482 |
| 1139 | NYSE:NOM | NOM | 未提供 | — |
| 1140 | NYSE:NOV | NOV | 未提供 | — |
| 1141 | NYSE:NOW | NOW | ServiceNow | — |
| 1142 | NYSE:NP | NP | 未提供 | — |
| 1143 | NYSE:NPK | NPK | 未提供 | — |
| 1144 | NYSE:NPN | NPN | 未提供 | — |
| 1145 | NYSE:NPO | NPO | 未提供 | Q16013760 |
| 1146 | NYSE:NPV | NPV | 未提供 | — |
| 1147 | NYSE:NQP | NQP | 未提供 | — |
| 1148 | NYSE:NRG | NRG | NRG Energy | — |
| 1149 | NYSE:NRK | NRK | 未提供 | — |
| 1150 | NYSE:NRP | NRP | 未提供 | — |
| 1151 | NYSE:NRT | NRT | 未提供 | — |
| 1152 | NYSE:NS | NS | 未提供 | Q7067712 |
| 1153 | NYSE:NSC | NSC | Norfolk Southern | Q20858163 |
| 1154 | NYSE:NSH | NSH | 未提供 | — |
| 1155 | NYSE:NSL | NSL | 未提供 | — |
| 1156 | NYSE:NSM | NSM | 未提供 | — |
| 1157 | NYSE:NSP | NSP | 未提供 | Q16980616 |
| 1158 | NYSE:NTC | NTC | 未提供 | — |
| 1159 | NYSE:NTG | NTG | 未提供 | — |
| 1160 | NYSE:NTL | NTL | 未提供 | — |
| 1161 | NYSE:NTX | NTX | 未提供 | — |
| 1162 | NYSE:NUE | NUE | Nucor | — |
| 1163 | NYSE:NUO | NUO | 未提供 | — |
| 1164 | NYSE:NUS | NUS | 未提供 | Q1764040 |
| 1165 | NYSE:NUV | NUV | 未提供 | — |
| 1166 | NYSE:NUW | NUW | 未提供 | — |
| 1167 | NYSE:NVG | NVG | 未提供 | — |
| 1168 | NYSE:NVR | NVR | NVR | Q6955706 |
| 1169 | NYSE:NVS | NVS | 未提供 | Q507154 |
| 1170 | NYSE:NWE | NWE | 未提供 | — |
| 1171 | NYSE:NWL | NWL | Newell Brands | — |
| 1172 | NYSE:NWN | NWN | 未提供 | Q6955989 |
| 1173 | NYSE:NX | NX | 未提供 | — |
| 1174 | NYSE:NXC | NXC | 未提供 | — |
| 1175 | NYSE:NXJ | NXJ | 未提供 | — |
| 1176 | NYSE:NXN | NXN | 未提供 | — |
| 1177 | NYSE:NXP | NXP | 未提供 | — |
| 1178 | NYSE:NXQ | NXQ | 未提供 | — |
| 1179 | NYSE:NXR | NXR | 未提供 | — |
| 1180 | NYSE:NYCB | NYCB | 未提供 | Q6517792 |
| 1181 | NYSE:NYT | NYT | 未提供 | Q2529982 |
| 1182 | NYSE:NYV | NYV | 未提供 | — |
| 1183 | NYSE:NZF | NZF | 未提供 | — |
| 1184 | NYSE:O | O | Realty Income Corporation | Q7301386 |
| 1185 | NYSE:OAK | OAK | 未提供 | — |
| 1186 | NYSE:OC | OC | 未提供 | Q7114736 |
| 1187 | NYSE:ODC | ODC | 未提供 | — |
| 1188 | NYSE:OFC | OFC | 未提供 | — |
| 1189 | NYSE:OGE | OGE | 未提供 | Q2007484 |
| 1190 | NYSE:OHI | OHI | 未提供 | — |
| 1191 | NYSE:OI | OI | 未提供 | Q180106 |
| 1192 | NYSE:OII | OII | 未提供 | Q1150535 |
| 1193 | NYSE:OIS | OIS | 未提供 | — |
| 1194 | NYSE:OKE | OKE | Oneok | Q11074156 |
| 1195 | NYSE:OLN | OLN | 未提供 | Q7086908 |
| 1196 | NYSE:OLP | OLP | 未提供 | — |
| 1197 | NYSE:OMC | OMC | Omnicom Group | — |
| 1198 | NYSE:OME | OME | 未提供 | — |
| 1199 | NYSE:OMI | OMI | 未提供 | — |
| 1200 | NYSE:OPY | OPY | 未提供 | Q7098552 |
| 1201 | NYSE:ORA | ORA | 未提供 | — |
| 1202 | NYSE:ORAN | ORAN | 未提供 | Q1431486 |
| 1203 | NYSE:ORCL | ORCL | Oracle | Q19900 |
| 1204 | NYSE:ORI | ORI | 未提供 | Q7084833 |
| 1205 | NYSE:OSK | OSK | 未提供 | Q1969091 |
| 1206 | NYSE:OXM | OXM | 未提供 | Q7115371 |
| 1207 | NYSE:OXY | OXY | Occidental Petroleum | Q2013357 |
| 1208 | NYSE:PAA | PAA | 未提供 | — |
| 1209 | NYSE:PAC | PAC | 未提供 | — |
| 1210 | NYSE:PAG | PAG | 未提供 | — |
| 1211 | NYSE:PAI | PAI | 未提供 | — |
| 1212 | NYSE:PANW | PANW | 未提供 | Q7128508 |
| 1213 | NYSE:PAY | PAY | 未提供 | Q2912154 |
| 1214 | NYSE:PB | PB | 未提供 | — |
| 1215 | NYSE:PBA | PBA | 未提供 | — |
| 1216 | NYSE:PBF | PBF | 未提供 | — |
| 1217 | NYSE:PBH | PBH | 未提供 | Q7241774 |
| 1218 | NYSE:PBI | PBI | 未提供 | — |
| 1219 | NYSE:PCF | PCF | 未提供 | — |
| 1220 | NYSE:PCG | PCG | 未提供 | Q1815011 |
| 1221 | NYSE:PCK | PCK | 未提供 | — |
| 1222 | NYSE:PCM | PCM | 未提供 | — |
| 1223 | NYSE:PCN | PCN | 未提供 | — |
| 1224 | NYSE:PCQ | PCQ | 未提供 | — |
| 1225 | NYSE:PDI | PDI | 未提供 | — |
| 1226 | NYSE:PDM | PDM | 未提供 | — |
| 1227 | NYSE:PDT | PDT | 未提供 | — |
| 1228 | NYSE:PEB | PEB | 未提供 | — |
| 1229 | NYSE:PEG | PEG | Public Service Enterprise Group | — |
| 1230 | NYSE:PEI | PEI | 未提供 | Q7164054 |
| 1231 | NYSE:PEI-B | PEI-B | 未提供 | — |
| 1232 | NYSE:PEO | PEO | 未提供 | Q7178935 |
| 1233 | NYSE:PEP | PEP | PepsiCo | Q334800 |
| 1234 | NYSE:PFD | PFD | 未提供 | — |
| 1235 | NYSE:PFE | PFE | Pfizer | Q206921 |
| 1236 | NYSE:PFG | PFG | Principal Financial Group | Q3403513 |
| 1237 | NYSE:PFH | PFH | 未提供 | — |
| 1238 | NYSE:PFK | PFK | 未提供 | — |
| 1239 | NYSE:PFL | PFL | 未提供 | — |
| 1240 | NYSE:PFN | PFN | 未提供 | — |
| 1241 | NYSE:PFO | PFO | 未提供 | — |
| 1242 | NYSE:PFS | PFS | 未提供 | Q7252517 |
| 1243 | NYSE:PG | PG | Procter & Gamble | Q212405 |
| 1244 | NYSE:PGP | PGP | 未提供 | — |
| 1245 | NYSE:PGR | PGR | Progressive Corporation | Q7248721 |
| 1246 | NYSE:PH | PH | Parker-Hannifin | — |
| 1247 | NYSE:PHD | PHD | 未提供 | — |
| 1248 | NYSE:PHG | PHG | 未提供 | Q170416 |
| 1249 | NYSE:PHH | PHH | 未提供 | Q7119231 |
| 1250 | NYSE:PHI | PHI | 未提供 | Q644167 |
| 1251 | NYSE:PHK | PHK | 未提供 | — |
| 1252 | NYSE:PHM | PHM | PulteGroup | — |
| 1253 | NYSE:PHT | PHT | 未提供 | — |
| 1254 | NYSE:PII | PII | 未提供 | Q1850587 |
| 1255 | NYSE:PJC | PJC | 未提供 | Q3389022 |
| 1256 | NYSE:PKE | PKE | 未提供 | — |
| 1257 | NYSE:PKG | PKG | Packaging Corporation of America | — |
| 1258 | NYSE:PKI | PKI | PerkinElmer | — |
| 1259 | NYSE:PKO | PKO | 未提供 | — |
| 1260 | NYSE:PKX | PKX | 未提供 | — |
| 1261 | NYSE:PLD | PLD | Prologis | — |
| 1262 | NYSE:PLOW | PLOW | 未提供 | — |
| 1263 | NYSE:PLT | PLT | 未提供 | — |
| 1264 | NYSE:PM | PM | Philip Morris International | Q1852556 |
| 1265 | NYSE:PMC | PMC | 未提供 | — |
| 1266 | NYSE:PMF | PMF | 未提供 | — |
| 1267 | NYSE:PML | PML | 未提供 | — |
| 1268 | NYSE:PMM | PMM | 未提供 | — |
| 1269 | NYSE:PMO | PMO | 未提供 | — |
| 1270 | NYSE:PMT | PMT | 未提供 | — |
| 1271 | NYSE:PMX | PMX | 未提供 | — |
| 1272 | NYSE:PNC | PNC | PNC Financial Services | Q38928 |
| 1273 | NYSE:PNC-P | PNC-P | 未提供 | — |
| 1274 | NYSE:PNC-Q | PNC-Q | 未提供 | — |
| 1275 | NYSE:PNF | PNF | 未提供 | — |
| 1276 | NYSE:PNI | PNI | 未提供 | — |
| 1277 | NYSE:PNM | PNM | 未提供 | — |
| 1278 | NYSE:PNW | PNW | Pinnacle West Capital | Q26110522 |
| 1279 | NYSE:POL | POL | 未提供 | — |
| 1280 | NYSE:POR | POR | 未提供 | Q7231939 |
| 1281 | NYSE:POST | POST | 未提供 | — |
| 1282 | NYSE:POT | POT | 未提供 | Q902908 |
| 1283 | NYSE:PPL | PPL | PPL | Q2044512 |
| 1284 | NYSE:PRA | PRA | 未提供 | — |
| 1285 | NYSE:PRI | PRI | 未提供 | Q7243481 |
| 1286 | NYSE:PRLB | PRLB | 未提供 | — |
| 1287 | NYSE:PRO | PRO | 未提供 | Q20858221 |
| 1288 | NYSE:PRU | PRU | Prudential Financial | Q2114414 |
| 1289 | NYSE:PSA | PSA | Public Storage | — |
| 1290 | NYSE:PSA-U | PSA-U | 未提供 | — |
| 1291 | NYSE:PSA-V | PSA-V | 未提供 | — |
| 1292 | NYSE:PSA-W | PSA-W | 未提供 | — |
| 1293 | NYSE:PSB | PSB | 未提供 | — |
| 1294 | NYSE:PSB-U | PSB-U | 未提供 | — |
| 1295 | NYSE:PSF | PSF | 未提供 | — |
| 1296 | NYSE:PSO | PSO | 未提供 | Q1916402 |
| 1297 | NYSE:PSX | PSX | Phillips 66 | Q1656230 |
| 1298 | NYSE:PTR | PTR | 未提供 | Q503182 |
| 1299 | NYSE:PTY | PTY | 未提供 | — |
| 1300 | NYSE:PUK | PUK | 未提供 | Q1374135 |
| 1301 | NYSE:PUK-A | PUK-A | 未提供 | — |
| 1302 | NYSE:PVH | PVH | PVH | Q1132272 |
| 1303 | NYSE:PWR | PWR | Quanta Services | — |
| 1304 | NYSE:PX | PX | 未提供 | Q902780 |
| 1305 | NYSE:PXD | PXD | Pioneer Natural Resources | Q4046987 |
| 1306 | NYSE:PYN | PYN | 未提供 | — |
| 1307 | NYSE:PYS | PYS | 未提供 | — |
| 1308 | NYSE:PZC | PZC | 未提供 | — |
| 1309 | NYSE:PZN | PZN | 未提供 | — |
| 1310 | NYSE:QEP | QEP | 未提供 | — |
| 1311 | NYSE:QUAD | QUAD | 未提供 | — |
| 1312 | NYSE:R | R | 未提供 | Q2178198 |
| 1313 | NYSE:RAS-A | RAS-A | 未提供 | — |
| 1314 | NYSE:RAS-B | RAS-B | 未提供 | — |
| 1315 | NYSE:RAS-C | RAS-C | 未提供 | — |
| 1316 | NYSE:RBA | RBA | 未提供 | — |
| 1317 | NYSE:RBC | RBC | 未提供 | — |
| 1318 | NYSE:RBS-S | RBS-S | 未提供 | — |
| 1319 | NYSE:RCI | RCI | 未提供 | Q165684 |
| 1320 | NYSE:RCL | RCL | Royal Caribbean Group | Q427881 |
| 1321 | NYSE:RCS | RCS | 未提供 | — |
| 1322 | NYSE:RDC | RDC | 未提供 | — |
| 1323 | NYSE:RDN | RDN | 未提供 | Q7280290 |
| 1324 | NYSE:RDS.A | RDS.A | 未提供 | Q154950 |
| 1325 | NYSE:RDS.B | RDS.B | 未提供 | — |
| 1326 | NYSE:RDY | RDY | 未提供 | Q908961 |
| 1327 | NYSE:RE | RE | Everest Re | — |
| 1328 | NYSE:REG | REG | Regency Centers | Q7308000 |
| 1329 | NYSE:RENN | RENN | 未提供 | — |
| 1330 | NYSE:RES | RES | 未提供 | — |
| 1331 | NYSE:RESI | RESI | 未提供 | — |
| 1332 | NYSE:REV | REV | 未提供 | Q1459595 |
| 1333 | NYSE:REX | REX | 未提供 | — |
| 1334 | NYSE:RF | RF | Regions Financial Corporation | Q917131 |
| 1335 | NYSE:RF-A | RF-A | 未提供 | — |
| 1336 | NYSE:RFI | RFI | 未提供 | — |
| 1337 | NYSE:RGA | RGA | 未提供 | — |
| 1338 | NYSE:RGC | RGC | 未提供 | Q835638 |
| 1339 | NYSE:RGR | RGR | 未提供 | Q1508526 |
| 1340 | NYSE:RGS | RGS | 未提供 | Q7309325 |
| 1341 | NYSE:RH | RH | 未提供 | — |
| 1342 | NYSE:RHI | RHI | Robert Half International | Q2157448 |
| 1343 | NYSE:RHP | RHP | 未提供 | — |
| 1344 | NYSE:RHT | RHT | 未提供 | Q485593 |
| 1345 | NYSE:RIG | RIG | 未提供 | Q686785 |
| 1346 | NYSE:RIO | RIO | 未提供 | Q821293 |
| 1347 | NYSE:RJF | RJF | Raymond James Financial | Q7298901 |
| 1348 | NYSE:RL | RL | Ralph Lauren Corporation | Q1641437 |
| 1349 | NYSE:RLGY | RLGY | 未提供 | — |
| 1350 | NYSE:RLH | RLH | 未提供 | Q7304530 |
| 1351 | NYSE:RLI | RLI | 未提供 | Q7276947 |
| 1352 | NYSE:RLJ | RLJ | 未提供 | — |
| 1353 | NYSE:RM | RM | 未提供 | — |
| 1354 | NYSE:RMD | RMD | ResMed | — |
| 1355 | NYSE:RMT | RMT | 未提供 | — |
| 1356 | NYSE:RNP | RNP | 未提供 | — |
| 1357 | NYSE:RNR | RNR | 未提供 | — |
| 1358 | NYSE:RNR-C | RNR-C | 未提供 | — |
| 1359 | NYSE:ROG | ROG | 未提供 | Q16984188 |
| 1360 | NYSE:ROK | ROK | Rockwell Automation | Q1529234 |
| 1361 | NYSE:ROL | ROL | Rollins | Q7361273 |
| 1362 | NYSE:ROP | ROP | Roper Technologies | Q16932392 |
| 1363 | NYSE:RPAI | RPAI | 未提供 | — |
| 1364 | NYSE:RPM | RPM | 未提供 | — |
| 1365 | NYSE:RPT | RPT | 未提供 | Q7289304 |
| 1366 | NYSE:RPT-D | RPT-D | 未提供 | — |
| 1367 | NYSE:RQI | RQI | 未提供 | — |
| 1368 | NYSE:RRC | RRC | 未提供 | Q4048013 |
| 1369 | NYSE:RRD | RRD | 未提供 | — |
| 1370 | NYSE:RRTS | RRTS | 未提供 | — |
| 1371 | NYSE:RS | RS | 未提供 | Q7311031 |
| 1372 | NYSE:RSG | RSG | Republic Services | Q7314415 |
| 1373 | NYSE:RSO | RSO | 未提供 | — |
| 1374 | NYSE:RSO-A | RSO-A | 未提供 | — |
| 1375 | NYSE:RSO-B | RSO-B | 未提供 | — |
| 1376 | NYSE:RST | RST | 未提供 | Q7368695 |
| 1377 | NYSE:RTEC | RTEC | 未提供 | — |
| 1378 | NYSE:RTN | RTN | 未提供 | Q156644 |
| 1379 | NYSE:RVT | RVT | 未提供 | — |
| 1380 | NYSE:RWT | RWT | 未提供 | — |
| 1381 | NYSE:RXN | RXN | 未提供 | — |
| 1382 | NYSE:RY | RY | 未提供 | Q735261 |
| 1383 | NYSE:RYN | RYN | 未提供 | — |
| 1384 | NYSE:SAH | SAH | 未提供 | — |
| 1385 | NYSE:SAM | SAM | 未提供 | Q894558 |
| 1386 | NYSE:SAN-A | SAN-A | 未提供 | — |
| 1387 | NYSE:SAN-B | SAN-B | 未提供 | — |
| 1388 | NYSE:SAN-C | SAN-C | 未提供 | — |
| 1389 | NYSE:SAN-I | SAN-I | 未提供 | — |
| 1390 | NYSE:SAP | SAP | 未提供 | Q552581 |
| 1391 | NYSE:SAR | SAR | 未提供 | — |
| 1392 | NYSE:SBH | SBH | 未提供 | Q7405065 |
| 1393 | NYSE:SBI | SBI | 未提供 | — |
| 1394 | NYSE:SBR | SBR | 未提供 | — |
| 1395 | NYSE:SCCO | SCCO | 未提供 | — |
| 1396 | NYSE:SCD | SCD | 未提供 | — |
| 1397 | NYSE:SCG | SCG | 未提供 | Q7389067 |
| 1398 | NYSE:SCHW | SCHW | Charles Schwab Corporation | — |
| 1399 | NYSE:SCI | SCI | 未提供 | Q11306218 |
| 1400 | NYSE:SCL | SCL | 未提供 | Q3208192 |
| 1401 | NYSE:SCM | SCM | 未提供 | — |
| 1402 | NYSE:SCS | SCS | 未提供 | — |
| 1403 | NYSE:SCX | SCX | 未提供 | — |
| 1404 | NYSE:SEE | SEE | Sealed Air | — |
| 1405 | NYSE:SEM | SEM | 未提供 | Q16984522 |
| 1406 | NYSE:SEMG | SEMG | 未提供 | — |
| 1407 | NYSE:SEP | SEP | 未提供 | — |
| 1408 | NYSE:SF | SF | 未提供 | — |
| 1409 | NYSE:SFE | SFE | 未提供 | — |
| 1410 | NYSE:SFL | SFL | 未提供 | — |
| 1411 | NYSE:SGF | SGF | 未提供 | — |
| 1412 | NYSE:SHG | SHG | 未提供 | Q487230 |
| 1413 | NYSE:SHI | SHI | 未提供 | — |
| 1414 | NYSE:SHO | SHO | 未提供 | — |
| 1415 | NYSE:SHW | SHW | Sherwin-Williams | Q48881 |
| 1416 | NYSE:SIG | SIG | 未提供 | — |
| 1417 | NYSE:SIX | SIX | 未提供 | Q824278 |
| 1418 | NYSE:SJM | SJM | JM Smucker | Q7742790 |
| 1419 | NYSE:SJR | SJR | 未提供 | — |
| 1420 | NYSE:SJW | SJW | 未提供 | Q17141393 |
| 1421 | NYSE:SKM | SKM | 未提供 | Q493145 |
| 1422 | NYSE:SKT | SKT | 未提供 | Q7682888 |
| 1423 | NYSE:SKX | SKX | 未提供 | Q2945643 |
| 1424 | NYSE:SLB | SLB | Schlumberger | Q1425316 |
| 1425 | NYSE:SLCA | SLCA | 未提供 | — |
| 1426 | NYSE:SLF | SLF | 未提供 | Q835456 |
| 1427 | NYSE:SLG | SLG | 未提供 | Q7391016 |
| 1428 | NYSE:SLG-I | SLG-I | 未提供 | — |
| 1429 | NYSE:SM | SM | 未提供 | Q16984687 |
| 1430 | NYSE:SMFG | SMFG | 未提供 | Q2094989 |
| 1431 | NYSE:SMG | SMG | 未提供 | Q2414097 |
| 1432 | NYSE:SMLP | SMLP | 未提供 | — |
| 1433 | NYSE:SMM | SMM | 未提供 | — |
| 1434 | NYSE:SMP | SMP | 未提供 | — |
| 1435 | NYSE:SNA | SNA | Snap-on | — |
| 1436 | NYSE:SNE | SNE | 未提供 | Q41187 |
| 1437 | NYSE:SNN | SNN | 未提供 | — |
| 1438 | NYSE:SNP | SNP | 未提供 | Q831445 |
| 1439 | NYSE:SNX | SNX | 未提供 | — |
| 1440 | NYSE:SNY | SNY | 未提供 | Q158205 |
| 1441 | NYSE:SO | SO | Southern Company | — |
| 1442 | NYSE:SON | SON | 未提供 | Q7562151 |
| 1443 | NYSE:SOR | SOR | 未提供 | — |
| 1444 | NYSE:SPA | SPA | 未提供 | — |
| 1445 | NYSE:SPB | SPB | 未提供 | — |
| 1446 | NYSE:SPE | SPE | 未提供 | — |
| 1447 | NYSE:SPG | SPG | Simon Property Group | Q2287759 |
| 1448 | NYSE:SPGI | SPGI | S&P Global | — |
| 1449 | NYSE:SPH | SPH | 未提供 | — |
| 1450 | NYSE:SPLP | SPLP | 未提供 | — |
| 1451 | NYSE:SPN | SPN | 未提供 | Q20486937 |
| 1452 | NYSE:SPR | SPR | 未提供 | — |
| 1453 | NYSE:SPXX | SPXX | 未提供 | — |
| 1454 | NYSE:SQM | SQM | 未提供 | — |
| 1455 | NYSE:SRE | SRE | Sempra Energy | — |
| 1456 | NYSE:SRF | SRF | 未提供 | — |
| 1457 | NYSE:SRI | SRI | 未提供 | — |
| 1458 | NYSE:SSD | SSD | 未提供 | — |
| 1459 | NYSE:SSL | SSL | 未提供 | — |
| 1460 | NYSE:SSP | SSP | 未提供 | — |
| 1461 | NYSE:SSTK | SSTK | 未提供 | — |
| 1462 | NYSE:SSW | SSW | 未提供 | — |
| 1463 | NYSE:SSW-D | SSW-D | 未提供 | — |
| 1464 | NYSE:ST | ST | 未提供 | — |
| 1465 | NYSE:STAG | STAG | 未提供 | — |
| 1466 | NYSE:STAR | STAR | 未提供 | — |
| 1467 | NYSE:STC | STC | 未提供 | — |
| 1468 | NYSE:STI | STI | 未提供 | Q181507 |
| 1469 | NYSE:STI-A | STI-A | 未提供 | — |
| 1470 | NYSE:STI-E | STI-E | 未提供 | — |
| 1471 | NYSE:STK | STK | 未提供 | — |
| 1472 | NYSE:STL | STL | 未提供 | — |
| 1473 | NYSE:STM | STM | 未提供 | Q661845 |
| 1474 | NYSE:STN | STN | 未提供 | — |
| 1475 | NYSE:STO | STO | 未提供 | Q1776022 |
| 1476 | NYSE:STON | STON | 未提供 | — |
| 1477 | NYSE:STT | STT | State Street Corporation | Q2037125 |
| 1478 | NYSE:STT-C | STT-C | 未提供 | — |
| 1479 | NYSE:STWD | STWD | 未提供 | — |
| 1480 | NYSE:STZ | STZ | Constellation Brands | — |
| 1481 | NYSE:STZ.B | STZ.B | 未提供 | — |
| 1482 | NYSE:SU | SU | 未提供 | Q1516410 |
| 1483 | NYSE:SUI | SUI | 未提供 | Q7638250 |
| 1484 | NYSE:SUI-A | SUI-A | 未提供 | — |
| 1485 | NYSE:SUN | SUN | 未提供 | — |
| 1486 | NYSE:SUP | SUP | 未提供 | — |
| 1487 | NYSE:SVU | SVU | 未提供 | — |
| 1488 | NYSE:SWK | SWK | Stanley Black & Decker | Q706094 |
| 1489 | NYSE:SWM | SWM | 未提供 | — |
| 1490 | NYSE:SWN | SWN | 未提供 | Q7571413 |
| 1491 | NYSE:SWX | SWX | 未提供 | Q7571280 |
| 1492 | NYSE:SWZ | SWZ | 未提供 | — |
| 1493 | NYSE:SXI | SXI | 未提供 | — |
| 1494 | NYSE:SXT | SXT | 未提供 | — |
| 1495 | NYSE:SYK | SYK | Stryker Corporation | — |
| 1496 | NYSE:SYT | SYT | 未提供 | Q688613 |
| 1497 | NYSE:SYX | SYX | 未提供 | Q7663801 |
| 1498 | NYSE:SYY | SYY | Sysco | Q2376711 |
| 1499 | NYSE:SZC | SZC | 未提供 | — |
| 1500 | NYSE:T | T | AT&T | Q35476 |
| 1501 | NYSE:TAP | TAP | Molson Coors Beverage Company | Q3319685 |
| 1502 | NYSE:TARO | TARO | 未提供 | Q16131515 |
| 1503 | NYSE:TBI | TBI | 未提供 | Q33127130 |
| 1504 | NYSE:TCAP | TCAP | 未提供 | — |
| 1505 | NYSE:TCF | TCF | 未提供 | — |
| 1506 | NYSE:TCF-C | TCF-C | 未提供 | — |
| 1507 | NYSE:TCO | TCO | 未提供 | Q7688650 |
| 1508 | NYSE:TCO-J | TCO-J | 未提供 | — |
| 1509 | NYSE:TCP | TCP | 未提供 | — |
| 1510 | NYSE:TD | TD | 未提供 | Q1194832 |
| 1511 | NYSE:TDC | TDC | 未提供 | Q430745 |
| 1512 | NYSE:TDE | TDE | 未提供 | — |
| 1513 | NYSE:TDF | TDF | 未提供 | — |
| 1514 | NYSE:TDG | TDG | TransDigm Group | Q7833612 |
| 1515 | NYSE:TDI | TDI | 未提供 | — |
| 1516 | NYSE:TDJ | TDJ | 未提供 | — |
| 1517 | NYSE:TDY | TDY | Teledyne Technologies | Q7696127 |
| 1518 | NYSE:TEF | TEF | 未提供 | — |
| 1519 | NYSE:TEI | TEI | 未提供 | — |
| 1520 | NYSE:TEL | TEL | TE Connectivity | Q663452 |
| 1521 | NYSE:TEN | TEN | 未提供 | — |
| 1522 | NYSE:TEO | TEO | 未提供 | Q1544312 |
| 1523 | NYSE:TER | TER | Teradyne | Q2404292 |
| 1524 | NYSE:TEVA | TEVA | 未提供 | Q898750 |
| 1525 | NYSE:TEX | TEX | 未提供 | Q1937503 |
| 1526 | NYSE:TFX | TFX | Teleflex | — |
| 1527 | NYSE:TG | TG | 未提供 | — |
| 1528 | NYSE:TGH | TGH | 未提供 | — |
| 1529 | NYSE:TGI | TGI | 未提供 | Q15507516 |
| 1530 | NYSE:TGNA | TGNA | 未提供 | Q20648663 |
| 1531 | NYSE:TGT | TGT | Target Corporation | Q1046951 |
| 1532 | NYSE:THC | THC | 未提供 | Q3518066 |
| 1533 | NYSE:THG | THG | 未提供 | Q5649395 |
| 1534 | NYSE:THO | THO | 未提供 | Q7796068 |
| 1535 | NYSE:THR | THR | 未提供 | — |
| 1536 | NYSE:THS | THS | 未提供 | Q16958009 |
| 1537 | NYSE:TI | TI | 未提供 | Q144617 |
| 1538 | NYSE:TIF | TIF | 未提供 | — |
| 1539 | NYSE:TISI | TISI | 未提供 | — |
| 1540 | NYSE:TJX | TJX | TJX Companies | Q1285819 |
| 1541 | NYSE:TKC | TKC | 未提供 | Q283852 |
| 1542 | NYSE:TKF | TKF | 未提供 | — |
| 1543 | NYSE:TKR | TKR | 未提供 | — |
| 1544 | NYSE:TLI | TLI | 未提供 | — |
| 1545 | NYSE:TLK | TLK | 未提供 | — |
| 1546 | NYSE:TLP | TLP | 未提供 | — |
| 1547 | NYSE:TLYS | TLYS | 未提供 | — |
| 1548 | NYSE:TM | TM | 未提供 | Q53268 |
| 1549 | NYSE:TMO | TMO | Thermo Fisher Scientific | — |
| 1550 | NYSE:TNC | TNC | 未提供 | — |
| 1551 | NYSE:TNH | TNH | 未提供 | — |
| 1552 | NYSE:TOL | TOL | 未提供 | — |
| 1553 | NYSE:TOT | TOT | 未提供 | Q154037 |
| 1554 | NYSE:TOWR | TOWR | 未提供 | — |
| 1555 | NYSE:TPC | TPC | 未提供 | Q7856971 |
| 1556 | NYSE:TPL | TPL | 未提供 | — |
| 1557 | NYSE:TPR | TPR | Tapestry | — |
| 1558 | NYSE:TPX | TPX | 未提供 | — |
| 1559 | NYSE:TPZ | TPZ | 未提供 | — |
| 1560 | NYSE:TR | TR | 未提供 | — |
| 1561 | NYSE:TRC | TRC | 未提供 | Q7695260 |
| 1562 | NYSE:TREC | TREC | 未提供 | — |
| 1563 | NYSE:TREX | TREX | 未提供 | — |
| 1564 | NYSE:TRGP | TRGP | 未提供 | — |
| 1565 | NYSE:TRI | TRI | 未提供 | Q1141267 |
| 1566 | NYSE:TRK | TRK | 未提供 | Q2308517 |
| 1567 | NYSE:TRN | TRN | 未提供 | — |
| 1568 | NYSE:TRNO | TRNO | 未提供 | — |
| 1569 | NYSE:TRP | TRP | 未提供 | Q1516108 |
| 1570 | NYSE:TRV | TRV | The Travelers Companies | — |
| 1571 | NYSE:TS | TS | 未提供 | — |
| 1572 | NYSE:TSI | TSI | 未提供 | — |
| 1573 | NYSE:TSM | TSM | 未提供 | Q713418 |
| 1574 | NYSE:TSN | TSN | Tyson Foods | Q1325987 |
| 1575 | NYSE:TSS | TSS | 未提供 | — |
| 1576 | NYSE:TSU | TSU | 未提供 | — |
| 1577 | NYSE:TTC | TTC | 未提供 | Q935059 |
| 1578 | NYSE:TTF | TTF | 未提供 | — |
| 1579 | NYSE:TTM | TTM | 未提供 | Q188514 |
| 1580 | NYSE:TTP | TTP | 未提供 | — |
| 1581 | NYSE:TU | TU | 未提供 | Q165858 |
| 1582 | NYSE:TUP | TUP | 未提供 | Q16985829 |
| 1583 | NYSE:TV | TV | 未提供 | Q47099 |
| 1584 | NYSE:TVC | TVC | 未提供 | — |
| 1585 | NYSE:TVE | TVE | 未提供 | — |
| 1586 | NYSE:TWN | TWN | 未提供 | — |
| 1587 | NYSE:TWO | TWO | 未提供 | — |
| 1588 | NYSE:TWX | TWX | 未提供 | Q191715 |
| 1589 | NYSE:TX | TX | 未提供 | — |
| 1590 | NYSE:TXT | TXT | Textron | Q2140315 |
| 1591 | NYSE:TY | TY | 未提供 | — |
| 1592 | NYSE:TYG | TYG | 未提供 | — |
| 1593 | NYSE:TYL | TYL | Tyler Technologies | — |
| 1594 | NYSE:UAA | UAA | Under Armour (Class A) | — |
| 1595 | NYSE:UAL | UAL | United Airlines | Q174769 |
| 1596 | NYSE:UBA | UBA | 未提供 | — |
| 1597 | NYSE:UBP | UBP | 未提供 | — |
| 1598 | NYSE:UDR | UDR | UDR | — |
| 1599 | NYSE:UFI | UFI | 未提供 | — |
| 1600 | NYSE:UFS | UFS | 未提供 | Q1238739 |
| 1601 | NYSE:UGI | UGI | 未提供 | — |
| 1602 | NYSE:UGP | UGP | 未提供 | — |
| 1603 | NYSE:UHS | UHS | Universal Health Services | Q7893967 |
| 1604 | NYSE:UHT | UHT | 未提供 | — |
| 1605 | NYSE:UIS | UIS | 未提供 | Q518016 |
| 1606 | NYSE:UL | UL | 未提供 | — |
| 1607 | NYSE:UMH | UMH | 未提供 | — |
| 1608 | NYSE:UN | UN | 未提供 | Q157062 |
| 1609 | NYSE:UNF | UNF | 未提供 | Q7884645 |
| 1610 | NYSE:UNH | UNH | UnitedHealth Group | — |
| 1611 | NYSE:UNM | UNM | 未提供 | — |
| 1612 | NYSE:UNP | UNP | Union Pacific | Q725793 |
| 1613 | NYSE:UPS | UPS | United Parcel Service | Q155026 |
| 1614 | NYSE:URI | URI | United Rentals | Q7889101 |
| 1615 | NYSE:USB | USB | U.S. Bancorp | Q739084 |
| 1616 | NYSE:USB-H | USB-H | 未提供 | — |
| 1617 | NYSE:USB-M | USB-M | 未提供 | — |
| 1618 | NYSE:USG | USG | 未提供 | — |
| 1619 | NYSE:USM | USM | 未提供 | Q2466256 |
| 1620 | NYSE:USNA | USNA | 未提供 | Q2502355 |
| 1621 | NYSE:USPH | USPH | 未提供 | — |
| 1622 | NYSE:UTF | UTF | 未提供 | — |
| 1623 | NYSE:UTL | UTL | 未提供 | Q17144598 |
| 1624 | NYSE:UTX | UTX | 未提供 | Q1045758 |
| 1625 | NYSE:UVV | UVV | 未提供 | Q7893929 |
| 1626 | NYSE:UZA | UZA | 未提供 | — |
| 1627 | NYSE:V | V | Visa | Q328840 |
| 1628 | NYSE:VAC | VAC | 未提供 | — |
| 1629 | NYSE:VAR | VAR | 未提供 | — |
| 1630 | NYSE:VBF | VBF | 未提供 | — |
| 1631 | NYSE:VC | VC | 未提供 | — |
| 1632 | NYSE:VCO | VCO | 未提供 | — |
| 1633 | NYSE:VCRA | VCRA | 未提供 | — |
| 1634 | NYSE:VER | VER | 未提供 | — |
| 1635 | NYSE:VET | VET | 未提供 | — |
| 1636 | NYSE:VFC | VFC | VF Corporation | — |
| 1637 | NYSE:VGI | VGI | 未提供 | — |
| 1638 | NYSE:VGM | VGM | 未提供 | — |
| 1639 | NYSE:VGR | VGR | 未提供 | — |
| 1640 | NYSE:VIV | VIV | 未提供 | — |
| 1641 | NYSE:VKQ | VKQ | 未提供 | — |
| 1642 | NYSE:VLO | VLO | Valero Energy | Q1283291 |
| 1643 | NYSE:VLT | VLT | 未提供 | — |
| 1644 | NYSE:VLY | VLY | 未提供 | — |
| 1645 | NYSE:VMC | VMC | Vulcan Materials | Q1685287 |
| 1646 | NYSE:VMI | VMI | 未提供 | Q7912506 |
| 1647 | NYSE:VMW | VMW | 未提供 | Q11407 |
| 1648 | NYSE:VNO | VNO | Vornado Realty Trust | Q368181 |
| 1649 | NYSE:VNO-I | VNO-I | 未提供 | — |
| 1650 | NYSE:VNO-K | VNO-K | 未提供 | — |
| 1651 | NYSE:VNTV | VNTV | 未提供 | — |
| 1652 | NYSE:VPG | VPG | 未提供 | — |
| 1653 | NYSE:VPV | VPV | 未提供 | — |
| 1654 | NYSE:VR | VR | 未提供 | — |
| 1655 | NYSE:VSH | VSH | 未提供 | — |
| 1656 | NYSE:VTA | VTA | 未提供 | — |
| 1657 | NYSE:VTR | VTR | Ventas | — |
| 1658 | NYSE:VVC | VVC | 未提供 | Q7917863 |
| 1659 | NYSE:VVI | VVI | 未提供 | — |
| 1660 | NYSE:VZ | VZ | Verizon Communications | Q467752 |
| 1661 | NYSE:WAB | WAB | Wabtec | — |
| 1662 | NYSE:WAGE | WAGE | 未提供 | — |
| 1663 | NYSE:WAIR | WAIR | 未提供 | — |
| 1664 | NYSE:WAL | WAL | 未提供 | — |
| 1665 | NYSE:WAT | WAT | Waters Corporation | Q1322011 |
| 1666 | NYSE:WBC | WBC | 未提供 | Q7946278 |
| 1667 | NYSE:WBK | WBK | 未提供 | — |
| 1668 | NYSE:WBS | WBS | 未提供 | Q7978891 |
| 1669 | NYSE:WCC | WCC | 未提供 | Q7949044 |
| 1670 | NYSE:WCG | WCG | 未提供 | Q7981065 |
| 1671 | NYSE:WCN | WCN | 未提供 | — |
| 1672 | NYSE:WD | WD | 未提供 | — |
| 1673 | NYSE:WDR | WDR | 未提供 | Q7958965 |
| 1674 | NYSE:WEA | WEA | 未提供 | — |
| 1675 | NYSE:WEC | WEC | WEC Energy Group | Q8027166 |
| 1676 | NYSE:WES | WES | 未提供 | — |
| 1677 | NYSE:WEX | WEX | 未提供 | Q8038212 |
| 1678 | NYSE:WFC | WFC | Wells Fargo | Q744149 |
| 1679 | NYSE:WFC-J | WFC-J | 未提供 | — |
| 1680 | NYSE:WFC-L | WFC-L | 未提供 | — |
| 1681 | NYSE:WFC-N | WFC-N | 未提供 | — |
| 1682 | NYSE:WFC-O | WFC-O | 未提供 | — |
| 1683 | NYSE:WGL | WGL | 未提供 | — |
| 1684 | NYSE:WGO | WGO | 未提供 | — |
| 1685 | NYSE:WGP | WGP | 未提供 | — |
| 1686 | NYSE:WHG | WHG | 未提供 | Q21010691 |
| 1687 | NYSE:WHR | WHR | Whirlpool Corporation | Q1471346 |
| 1688 | NYSE:WIA | WIA | 未提供 | — |
| 1689 | NYSE:WIW | WIW | 未提供 | — |
| 1690 | NYSE:WLK | WLK | 未提供 | Q17326957 |
| 1691 | NYSE:WLL | WLL | 未提供 | — |
| 1692 | NYSE:WM | WM | Waste Management | Q2552219 |
| 1693 | NYSE:WMB | WMB | Williams Companies | Q930639 |
| 1694 | NYSE:WMC | WMC | 未提供 | — |
| 1695 | NYSE:WMK | WMK | 未提供 | — |
| 1696 | NYSE:WMT | WMT | Walmart | Q483551 |
| 1697 | NYSE:WNC | WNC | 未提供 | — |
| 1698 | NYSE:WNS | WNS | 未提供 | Q7953597 |
| 1699 | NYSE:WOR | WOR | 未提供 | Q2594130 |
| 1700 | NYSE:WPC | WPC | 未提供 | — |
| 1701 | NYSE:WPM | WPM | 未提供 | — |
| 1702 | NYSE:WPZ | WPZ | 未提供 | — |
| 1703 | NYSE:WR | WR | 未提供 | — |
| 1704 | NYSE:WRB | WRB | W. R. Berkley Corporation | — |
| 1705 | NYSE:WRE | WRE | 未提供 | — |
| 1706 | NYSE:WRI | WRI | 未提供 | Q17091323 |
| 1707 | NYSE:WSM | WSM | 未提供 | Q2581220 |
| 1708 | NYSE:WSO | WSO | 未提供 | — |
| 1709 | NYSE:WSO.B | WSO.B | 未提供 | — |
| 1710 | NYSE:WSR | WSR | 未提供 | — |
| 1711 | NYSE:WST | WST | West Pharmaceutical Services | — |
| 1712 | NYSE:WTM | WTM | 未提供 | — |
| 1713 | NYSE:WTR | WTR | 未提供 | — |
| 1714 | NYSE:WTS | WTS | 未提供 | — |
| 1715 | NYSE:WU | WU | Western Union | Q861042 |
| 1716 | NYSE:WWE | WWE | 未提供 | Q35339 |
| 1717 | NYSE:WWW | WWW | 未提供 | Q8030548 |
| 1718 | NYSE:WY | WY | Weyerhaeuser | Q1780382 |
| 1719 | NYSE:WYN | WYN | 未提供 | Q303341 |
| 1720 | NYSE:X | X | 未提供 | Q72539 |
| 1721 | NYSE:XEC | XEC | 未提供 | — |
| 1722 | NYSE:XEL | XEL | Xcel Energy | — |
| 1723 | NYSE:XOM | XOM | ExxonMobil | Q156238 |
| 1724 | NYSE:XOXO | XOXO | 未提供 | — |
| 1725 | NYSE:XPO | XPO | 未提供 | — |
| 1726 | NYSE:XRX | XRX | 未提供 | Q152433 |
| 1727 | NYSE:XYL | XYL | Xylem | Q3146975 |
| 1728 | NYSE:Y | Y | 未提供 | Q151670 |
| 1729 | NYSE:YELP | YELP | 未提供 | — |
| 1730 | NYSE:YPF | YPF | 未提供 | — |
| 1731 | NYSE:YUM | YUM | Yum! Brands | Q191615 |
| 1732 | NYSE:ZB-A | ZB-A | 未提供 | — |
| 1733 | NYSE:ZBH | ZBH | Zimmer Biomet | Q203552 |
| 1734 | NYSE:ZF | ZF | 未提供 | — |
| 1735 | NYSE:ZNH | ZNH | 未提供 | Q291090 |
| 1736 | NYSE:ZTR | ZTR | 未提供 | — |
