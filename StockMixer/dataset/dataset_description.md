# StockMixer 数据集说明

> 本文档对 `StockMixer/dataset/` 目录下所有数据文件的格式、字段含义及样例进行详细说明。

---

## 目录结构

```
dataset/
├── NASDAQ/
│   ├── eod_data.pkl      # EOD (End-of-Day) 多维特征数据
│   ├── gt_data.pkl       # 地面真值 (Ground Truth) — 收益率
│   ├── mask_data.pkl     # 有效数据掩码
│   └── price_data.pkl    # 收盘价数据 (close price)
├── NYSE/
│   ├── eod_data.pkl      # float32，(1737, 1245, 5)
│   ├── gt_data.pkl       # float32，(1737, 1245)，一步收益率
│   ├── mask_data.pkl     # float32，(1737, 1245)，有效掩码
│   ├── price_data.pkl    # float32，(1737, 1245)，归一化收盘价
│   ├── tickers.csv       # 股票轴顺序
│   ├── trading_dates.csv # 日期轴顺序
│   └── validation_report.json # 来源、哈希和检查结果
└── SP500/
    ├── SP500.npy               # 归一化后的 EOD 特征数据
    ├── baseline_data_sp500.npy # 原始 (未归一化) 股票数据
    └── sp500_ticker.csv        # 股票代码与行业信息
```

---

## 一、NASDAQ 数据集

### 1.1 eod_data.pkl

| 属性     | 值                          |
|----------|-----------------------------|
| 类型     | `numpy.ndarray` (float32)   |
| 形状     | `(1026, 1245, 5)`           |
| 含义     | 1026 只股票 × 1245 个交易日 × 5 个特征 |

**5 个特征维度（已归一化至 [0, 1]）：**

| 维度索引 | 特征名       | 说明                                                         |
|----------|-------------|--------------------------------------------------------------|
| 0        | `open`      | 开盘价（归一化）                                             |
| 1        | `high`      | 最高价（归一化）                                             |
| 2        | `low`       | 最低价（归一化）                                             |
| 3        | `volume`    | 成交量（归一化，与价格特征分布相似，原始数据源已做预处理）     |
| 4        | `close`     | 收盘价（归一化），**等于 `price_data`**                       |

> **缺失值处理**：原始数据中缺失值标记为 `-1234`，被替换为 `1.1`。

**样例**（股票 0，第 0~2 天）：

```
Day 0: [open=0.270533, high=0.269522, low=0.267237, volume=0.263333, close=0.275333]
Day 1: [open=0.271109, high=0.269741, low=0.267820, volume=0.263982, close=0.271219]
Day 2: [open=0.271822, high=0.270097, low=0.268485, volume=0.264723, close=0.272316]
```

---

### 1.2 gt_data.pkl（Ground Truth — 收益率）

| 属性     | 值                          |
|----------|------------------------------|
| 类型     | `numpy.ndarray` (float32)    |
| 形状     | `(1026, 1245)`               |
| 含义     | 每只股票每个交易日的收益率标签 |

**计算公式**：

```
gt[stock, t] = (close[t] - close[t - steps]) / close[t - steps]
```

其中 `steps = 1`（默认），即当日相对于前一交易日的**收盘价收益率**。

- 第 0 天：`gt = 0.0`（无前日数据）
- 缺失数据位置（`mask = 0`）：`gt = 0.0`

**值域**：`[-0.939, 0.982]`

**样例**（股票 0，第 0~4 天）：

```
Day 0:  0.000000    (首日)
Day 1: -0.014942    (下跌约 1.49%)
Day 2:  0.004045    (上涨约 0.40%)
Day 3: -0.023161    (下跌约 2.32%)
Day 4:  0.013402    (上涨约 1.34%)
```

---

### 1.3 mask_data.pkl（有效数据掩码）

| 属性     | 值                          |
|----------|------------------------------|
| 类型     | `numpy.ndarray` (float32)    |
| 形状     | `(1026, 1245)`               |
| 含义     | 标识对应位置的数据是否有效     |

| 值  | 含义                          |
|-----|-------------------------------|
| 1.0 | 数据有效                      |
| 0.0 | 数据缺失（该股票该日无交易数据）|

**统计**：有效数据占比 99.86%，缺失 1728 个位置。

**样例**（股票 0，第 0~4 天）：

```
[1.0, 1.0, 1.0, 1.0, 1.0]  ← 全部有效
```

---

### 1.4 price_data.pkl（收盘价）

| 属性     | 值                          |
|----------|------------------------------|
| 类型     | `numpy.ndarray` (float32)    |
| 形状     | `(1026, 1245)`               |
| 含义     | 每只股票每个交易日的归一化收盘价 |

**与 `eod_data` 的关系**：

```python
price_data == eod_data[:, :, 4]   # 完全一致
```

即 `price_data` 是 `eod_data` 第 4 维（close）的提取，方便训练时直接使用。

**值域**：`[0.035, 1.0]`

**样例**（股票 0，第 0~4 天）：

```
[0.275333, 0.271219, 0.272316, 0.266009, 0.269574]
```

---

## 二、NYSE 数据集

NYSE 数据已从本地 `Temporal_Relational_Stock_Ranking-master/data/2013-01-01` 的 processed CSV 生成，调用当前项目未经修改的 `src/load_data.py::load_EOD_data(..., steps=1)`。

- 严格沿用 `NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv` 中 1737 只股票的顺序，忽略源目录额外 32 个 CSV。
- 覆盖 2013-01-02 至 2017-12-08，共 1245 个交易日；原日期文件前 29 天为均线预热期，不进入输出。
- `eod_data` 五列依次为 **5 日、10 日、20 日、30 日收盘均线和收盘价**，不是 OHLCV；这些特征已经由上游按最大收盘价归一化，不再二次归一化。
- 四个数组均为 `float32`；`price_data == eod_data[:, :, -1]`。缺失标记 `-1234` 按原加载器替换为 `1.1`，以掩码排除相关训练窗口。
- 缺失股票日为 1389 个，有效比例约 99.9358%；有效相邻日期的标签与原始收盘价收益率计算完全一致。
- 名义训练段为 2013-01-02~2015-12-31，验证段为 2016-01-04~2016-12-30，测试段为 2017-01-03~2017-12-08。当前原始采样约定可能训练到验证期前 16 天。

重建命令（在项目根目录、stockmixer 环境执行）：

```powershell
python scripts/prepare_nyse_data.py --source-root "D:\finance\model\StockMixer\Temporal_Relational_Stock_Ranking-master"
```

脚本先检查全部源 CSV、生成并校验临时 pickle，再备份和替换目标文件。原文件备份在 `dataset_backups/NYSE_<时间戳>/`；来源哈希、数组哈希、形状和校验统计见 `NYSE/validation_report.json`。股票和日期映射分别见 `NYSE/tickers.csv`、`NYSE/trading_dates.csv`。

---

## 三、SP500 数据集

### 3.1 SP500.npy（归一化 EOD 特征）

| 属性     | 值                          |
|----------|------------------------------|
| 类型     | `numpy.ndarray` (float64)    |
| 形状     | `(474, 2526, 5)`             |
| 含义     | 474 只股票 × 2526 个交易日 × 5 个特征 |

**5 个特征维度（归一化至 [0, 1]）：**

| 维度索引 | 特征名       | 说明                              |
|----------|-------------|-----------------------------------|
| 0        | `open`      | 开盘价（归一化）                  |
| 1        | `high`      | 最高价（归一化）                  |
| 2        | `low`       | 最低价（归一化）                  |
| 3        | `volume`    | 成交量（归一化）                  |
| 4        | `close`     | 收盘价（归一化），用于计算收益率   |

> **注意**：`train.py` 中对 SP500 数据做了时间截取：`data = data[:, 915:, :]`，即实际使用第 915 天之后的数据（剩余 1611 个交易日）。

**样例**（股票 0 (POOL)，第 0~2 天；实际股票轴顺序见 `../docs/stock_identity_mapping.md`）：

```
Day 0: [open=0.0625, high=0.0633, low=0.0624, volume=0.0284, close=0.0628]
Day 1: [open=0.0621, high=0.0638, low=0.0627, volume=0.0404, close=0.0636]
Day 2: [open=0.0635, high=0.0637, low=0.0634, volume=0.0271, close=0.0634]
```

---

### 3.2 baseline_data_sp500.npy（原始未归一化数据）

| 属性     | 值                              |
|----------|----------------------------------|
| 类型     | `numpy.ndarray` (object, 0-dim)  |
| 内容     | `dict` — 键为股票代码，值为 `pandas.DataFrame` |
| 股票数量 | 474                              |

**DataFrame 列说明**：

| 列名       | 类型     | 说明           |
|------------|----------|----------------|
| `close`    | float64  | 收盘价（原始）  |
| `open`     | float64  | 开盘价（原始）  |
| `high`     | float64  | 最高价（原始）  |
| `low`      | float64  | 最低价（原始）  |
| `volume`   | float64  | 成交量（原始）  |
| `adjclose` | float64  | 复权收盘价      |

**日期范围**：2012-05-14 至 2022-05-25（2526 个交易日）

**样例**（POOL 股票）：

```
                         close       open       high        low    volume   adjclose
2012-05-14       36.299999  36.360001  36.840000  35.979999  230900.0  32.237789
2012-05-15       36.750000  36.139999  37.119999  36.139999  328500.0  32.637432
2012-05-16       36.619999  36.950001  37.080002  36.549999  220300.0  32.521984
```

---

### 3.3 sp500_ticker.csv（股票信息）

| 属性     | 值              |
|----------|-----------------|
| 行数     | 505（+ 1 行表头）|
| 编码     | UTF-8           |

**列说明**：

| 列名     | 说明           | 样例                   |
|----------|---------------|------------------------|
| `Symbol` | 股票代码       | `MMM`, `AOS`, `ABT`    |
| `Name`   | 公司全称       | `3M`, `A. O. Smith`    |
| `Sector` | 所属行业       | `Industrials`, `Health Care` |

**样例**：

```csv
Symbol,Name,Sector
MMM,3M,Industrials
AOS,A. O. Smith,Industrials
ABT,Abbott Laboratories,Health Care
ABBV,AbbVie,Health Care
```

---

## 四、数据来源

| 数据集    | 来源                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------|
| NASDAQ    | [Temporal Relational Ranking for Stock Prediction](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) |
| NYSE      | 同上；已从本地 processed CSV 重建，来源及校验见 `NYSE/validation_report.json`                            |
| SP500     | [Efficient Integration of Multi-Order Dynamics](https://github.com/thanhtrunghuynh93/estimate)                     |

---

## 五、训练时的数据划分

`train.py` 中的划分方式：

| 数据集  | 训练集            | 验证集              | 测试集                |
|---------|-------------------|---------------------|-----------------------|
| NASDAQ  | Day 0 ~ 755       | Day 756 ~ 1007      | Day 1008 ~ 1244       |
| NYSE    | Day 0 ~ 755       | Day 756 ~ 1007      | Day 1008 ~ 1244       |
| SP500   | Day 915 ~ 1670    | Day 1671 ~ 1922     | Day 1923 ~ 2525       |

> SP500 使用 `data[:, 915:, :]` 截取后，相当于原始日期从 915 天后开始计算。

---

## 六、数据加载代码逻辑

源自 `src/load_data.py`，NASDAQ/NYSE 的数据处理流程：

```
原始 CSV (每只股票一个文件)
  └─ np.genfromtxt 加载 → single_EOD (days × 6 列)
       ├─ 第 0 列: 交易日索引 (加载后跳过)
       ├─ 第 1~5 列: [close_ma5, close_ma10, close_ma20, close_ma30, close] (已归一化)
       └─ 缺失值标记: -1234 → 替换为 1.1

eod_data   = single_EOD[:, 1:]           # 跳过日期列 → (stocks, days, 5)
price_data = eod_data[:, :, 4]           # 最后一维 = close
mask_data  = 1.0 / 0.0                  # 基于缺失值判断
gt_data    = (close[t] - close[t-1]) / close[t-1]  # 收益率
```

---

## 七、注意事项

1. **NYSE 数据已补齐**：当前标签按 `steps=1` 生成；修改预测步长时需重新生成相应收益率标签。数据轴顺序、哈希和校验结果保存在 NYSE 目录的附属文件中。
2. **缺失值**：NASDAQ 数据中缺失值已替换为 `1.1`（超出正常值域 [0, 1]），训练时通过 `mask_data` 过滤。
3. **归一化方式**：NASDAQ/SP500 的归一化方式不同（来自不同数据源），请勿混合使用。
4. **SP500 截取**：`train.py` 运行时对 SP500 数据从第 915 天截取，`SP500.npy` 本身包含完整 2526 天。
