"""Verify stock-axis identities and export Markdown/CSV for text-data alignment."""

import csv
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "Temporal_Relational_Stock_Ranking-master" / "data"
OUT = ROOT / "docs"
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main():
    names_path = ROOT / "dataset/SP500/sp500_ticker.csv"
    names = pd.read_csv(names_path, dtype=str).fillna("")
    require(names.Symbol.is_unique, "Duplicate ticker symbols in company-name metadata.")
    names_by_symbol = names.set_index("Symbol").to_dict("index")
    rows = []
    checks = {}
    sources = {str(names_path): digest(names_path)}

    for dataset in ["NASDAQ", "SP500", "NYSE"]:
        print(f"Verifying {dataset} stock-axis identities...", flush=True)
        if dataset == "SP500":
            array_path = ROOT / "dataset/SP500/SP500.npy"
            raw_path = ROOT / "dataset/SP500/baseline_data_sp500.npy"
            array = np.load(array_path)
            raw = np.load(raw_path, allow_pickle=True).item()
            tickers = list(raw.keys())
            require(len(tickers) == 474 and len(set(tickers)) == 474, "Unexpected SP500 stock keys.")
            require(array.shape == (474, 2526, 5), "Unexpected SP500 shape.")
            # Verify the inferred dictionary-key order against every feature value.
            raw_features = np.stack([
                raw[ticker][["open", "high", "low", "volume", "close"]].to_numpy(dtype=float)
                for ticker in tickers
            ])
            expected = raw_features / raw_features.max(axis=1, keepdims=True)
            require(np.array_equal(array, expected), "SP500 dictionary order does not match the normalized tensor.")
            reference_dates = pd.to_datetime(raw[tickers[0]].index)
            require(all(pd.to_datetime(raw[t].index).equals(reference_dates) for t in tickers),
                    "SP500 stocks have different date axes.")
            require(all(t in names_by_symbol for t in tickers), "SP500 company-name metadata is incomplete.")
            wiki = {}
            dates = reference_dates.strftime("%Y-%m-%d").tolist()
            checks[dataset] = {
                "verification": "All normalized features equal raw features / per-stock per-feature maximum in dictionary-key order.",
                "max_absolute_error": 0.0,
                "training_start_raw_index": 915,
                "training_date_range": [dates[915], dates[-1]],
                "training_days": len(dates) - 915,
            }
            sources[str(raw_path)] = digest(raw_path)
            del raw_features, expected, raw
        else:
            array_path = ROOT / "dataset" / dataset / "eod_data.pkl"
            with array_path.open("rb") as stream:
                array = pickle.load(stream)
            ticker_path = SOURCE / f"{dataset}_tickers_qualify_dr-0.98_min-5_smooth.csv"
            tickers = np.genfromtxt(ticker_path, dtype=str, delimiter="\t").tolist()
            count = 1026 if dataset == "NASDAQ" else 1737
            require(len(tickers) == count and len(set(tickers)) == count, f"Unexpected {dataset} ticker list.")
            require(array.shape == (count, 1245, 5), f"Unexpected {dataset} shape.")
            for index, ticker in enumerate(tickers):
                path = SOURCE / "2013-01-01" / f"{dataset}_{ticker}_1.csv"
                raw = np.genfromtxt(path, dtype=np.float32, delimiter=",")
                if dataset == "NASDAQ":
                    raw = raw[:-1, :]
                require(raw.shape == (1245, 6) and np.array_equal(raw[:, 0], np.arange(1245)),
                        f"Unexpected source date axis: {path}")
                expected = raw[:, 1:].copy()
                expected[np.abs(expected + 1234) < 1e-8] = np.float32(1.1)
                require(np.array_equal(array[index], expected),
                        f"Stock identity mismatch: {dataset}[{index}] != {ticker}")
            wiki_path = SOURCE / f"{dataset}_wiki.csv"
            with wiki_path.open(encoding="utf-8", newline="") as stream:
                wiki_rows = list(csv.reader(stream))
            require([row[0] for row in wiki_rows] == tickers, f"{dataset} Wikidata ticker order mismatch.")
            wiki = dict(wiki_rows)
            date_path = SOURCE / f"{dataset}_aver_line_dates.csv"
            dates = [line[:10] for line in date_path.read_text(encoding="utf-8").splitlines()[29:]]
            if dataset == "NASDAQ":
                dates = dates[:-1]
            require(len(dates) == array.shape[1], f"{dataset} date-axis length mismatch.")
            if dataset == "NYSE":
                with (ROOT / "dataset/NYSE/tickers.csv").open(encoding="utf-8", newline="") as stream:
                    saved = [row["ticker"] for row in csv.DictReader(stream)]
                require(saved == tickers, "NYSE installed ticker metadata mismatch.")
            checks[dataset] = {
                "verification": "Every tensor row equals the qualified ticker's processed CSV after the unchanged loader's slicing and missing-value replacement.",
                "verified_source_csv_rows": len(tickers),
                "max_absolute_error": 0.0,
            }
            for path in [ticker_path, wiki_path, date_path]:
                sources[str(path)] = digest(path)

        for index, ticker in enumerate(tickers):
            company = names_by_symbol.get(ticker, {})
            qid = wiki.get(ticker, "")
            if qid == "unknown":
                qid = ""
            rows.append({
                "dataset": dataset,
                "array_index": index,
                "stock_uid": f"{dataset}:{ticker}",
                "ticker": ticker,
                "company_name": company.get("Name", ""),
                "company_name_status": "local_SP500_snapshot_exact_symbol_match" if company else "not_provided_locally",
                "sector_snapshot": company.get("Sector", ""),
                "wikidata_id": qid,
                "wikidata_id_status": "source_provided" if qid else ("not_provided_for_SP500" if dataset == "SP500" else "source_unknown"),
                "date_start": dates[0],
                "date_end": dates[-1],
            })
        checks[dataset].update({
            "stocks": len(tickers), "shape": list(array.shape), "dtype": str(array.dtype),
            "date_range": [dates[0], dates[-1]],
            "company_names_available": sum(bool(r["company_name"]) for r in rows if r["dataset"] == dataset),
            "source_wikidata_ids_available": sum(bool(r["wikidata_id"]) for r in rows if r["dataset"] == dataset),
        })
        sources[str(array_path)] = digest(array_path)

    require(len(rows) == 3237 and len({r["stock_uid"] for r in rows}) == 3237, "Incomplete or duplicate mapping.")
    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "stock_identity_mapping.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    text = ["""# StockMixer 股票身份映射与自然语言数据对齐说明

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
"""]
    for dataset in ["NASDAQ", "SP500", "NYSE"]:
        c = checks[dataset]
        text.append(f"| {dataset} | {c['stocks']} | {c['company_names_available']} | {c['stocks']-c['company_names_available']} | {c['source_wikidata_ids_available']} |\n")
    text.append("""
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

`published_at`、`available_at` 和 `prediction_cutoff` 在真实训练样本中应填为带时区的时间，并要求 `available_at <= prediction_cutoff`。区分新闻首发时间和你实际可获得该信息的时间。不要将目标日收盘后新闻或未来标签写入预测该日的输入；模型生成的模拟市场文本应使用 `is_synthetic=true` 并记录生成来源。

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
- 当前实现：`src/load_data.py`、`src/train.py`、`src/model.py`。主干不接收名称或 QID，本文提供的是文本扩展的对齐元数据，不修改训练逻辑或数据数组。
- [机器可读 CSV](stock_identity_mapping.csv)：完整 3237 行，包含名称来源状态、QID 状态、数据日期范围。
- [核验清单与输入文件 SHA256](stock_identity_mapping_validation.json)：用于确认该映射属于哪一版数组。
- 复现：在项目根目录的 stockmixer 环境运行 `python scripts/export_stock_identity_mapping.py`。

后续更换数据文件、股票集合或重排股票轴后，应重新生成并核验本映射；不得沿用旧行号。

## 6. 完整索引—股票映射表

以下 `array_index` 均从 0 开始；“建议 stock_uid”是新增连接键。公司名来自本地 SP500 名称快照，未提供处仍可直接使用 ticker 对齐；“—”表示无可用的源 QID。
""")
    for dataset in ["NASDAQ", "SP500", "NYSE"]:
        subset = [r for r in rows if r["dataset"] == dataset]
        text.append(f"\n### {dataset}：{len(subset)} 只\n\n")
        text.append("| array_index | 建议 stock_uid | 股票代码 ticker | 公司名称（本地快照） | 源 Wikidata QID |\n|---:|---|---|---|---|\n")
        for row in subset:
            cells = [row["array_index"], row["stock_uid"], row["ticker"], row["company_name"] or "未提供", row["wikidata_id"] or "—"]
            text.append("| " + " | ".join(escape(c) for c in cells) + " |\n")
    md_path = OUT / "stock_identity_mapping.md"
    md_path.write_text("".join(text), encoding="utf-8")
    manifest = {"datasets": checks, "total_mapping_rows": len(rows), "input_sha256": sources,
                "outputs": {csv_path.name: digest(csv_path), md_path.name: digest(md_path)}}
    (OUT / "stock_identity_mapping_validation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mapping_rows": len(rows), "datasets": checks, "markdown": str(md_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
