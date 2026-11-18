# 市场通道专项（M1–M6）操作说明

本文对应 `新闻融合现状_问题与后续实施方案_20261117.md` 第 7 节的市场通道专项。所有步骤都只涉及离线新闻增强、审核、路由和面板，不改动 StockMixer 主干、行情数组、标签或评估函数。训练循环仍只读取预计算面板。

解释器：`C:\Users\marti\anaconda3\envs\stockmixer\python.exe`（下文记为 `$py`）。

## 新增文件

| 文件 | 作用 |
|---|---|
| `news_fusion/market_review_completion.py` | M1：只补审核 `unreviewed_*` 记录，继承决定逐条重放校验 |
| `news_fusion/market_quality.py` | M2 证据来源分级、M5 事件聚类、M6 门槛、审核漏斗 |
| `news_fusion/market_evaluation.py` | M3：分层抽样、人工标注校验、精确率/召回率估计 |
| `news_fusion/official_events.py` | M4：官方宏观发布的校验、编码和规则路由 |
| `scripts/12_audit_market_channel.py` | 漏斗报告；`--evaluation-sample` 生成评估样本；`--score` 评分 |
| `scripts/13_complete_market_review.py` | M1 入口；`--check` 只统计不调用模型 |
| `scripts/14_build_official_events.py` | M4 入口 |
| `tests_news/test_market_channel.py` | 合成夹具契约测试 |

已修改：`panel.py`（`build_panel(..., market_channel=None)`）、`pipeline.py`（接受 M1 产物、传入 `market_channel`、M6 门槛）、`routing.py`（面板证明包含官方路由）、`review_reuse.py`、`configuration.py`、`inputs.py`（保留 `summary_origin`、`source_type`、`source_url`、`event_type`）。未配置 `market_channel` 时，这些路径的行为与修改前相同。

## 推荐顺序

### 1. 漏斗报告（只读）

```powershell
& $py scripts/12_audit_market_channel.py
```

输出 `outputs/reports/<时间>_NASDAQ_NYSE_SP500_market_channel_audit/market_funnel.json`，按数据集和切分统计 include/exclude/pending、审核方式、上游候选、证据来源和推测性摘要数量。切分按发布日期近似（只有日期的记录顺延到下一交易日），正式数字以面板 `market_quality` 为准。

### 2. M1 补审核

```powershell
& $py scripts/13_complete_market_review.py --check   # 每个数据集待补审核的条数
ollama list                                           # 确认 qwen3:8b 的 digest 与 llm.yaml 一致
& $py scripts/13_complete_market_review.py
```

- 第一个数据集调用本地 Qwen，其余数据集命中逐文章缓存，不重复请求。
- 输出 `outputs/artifacts/news/routing_reviews/<时间>_<DATASET>_market_review_v3/`，内含来源审核和本次运行的完整归档，`validate_completed_review` 会逐条重放。
- `reviewed_at` 是本次真实审核完成时间。
- 完成后把三个输出路径填入 `pipeline.review_artifacts`，再执行 `main.py --new` 新建准备版本。

### 3. M3 人工评估集

```powershell
& $py scripts/12_audit_market_channel.py --evaluation-sample --per-stratum 40
```

在 `outputs/artifacts/news/market_review_evaluation/<时间>_US_market_evaluation_sample/` 生成 `sample.jsonl` 和 `guideline.md`。复制 `sample.jsonl`，由人工填写 `human_label`（market / not_market / uncertain）、`evidence_quote`、`event_type`、`reviewed_by`、`reviewed_at`。标注时应隐藏 `model_decision` 和 `model_confidence` 两列，避免锚定。

```powershell
& $py scripts/12_audit_market_channel.py --score <样本目录> --labels <填写后的副本.jsonl>
```

报告 include 精确率、候选召回估计（分层外推）、按 `text_origin` 和置信度分箱的结果。审核者标识不能是模型或规则名称。修改 Prompt 或阈值时只在同一评估集上比较。

### 4. M4 官方宏观事件

上游 U 负责采集。每行 JSONL 至少包含：

```json
{"source_id": "official:fomc:2016-12-14", "source_type": "official_macro_release",
 "event_type": "fomc_statement", "title": "...", "text": "首次发布原文",
 "source_url": "https://www.federalreserve.gov/...", "published_at": "2016-12-14T14:00:00-05:00",
 "fetched_at": "...", "content_sha256": "text 的 SHA256", "first_release": true,
 "publication_time_basis": "official_page_timestamp"}
```

支持的 `event_type` 及其域名：`fomc_statement` / `fomc_minutes` → federalreserve.gov；`cpi` / `employment_situation` → bls.gov；`gdp` / `personal_income_outlays` → bea.gov。必须使用 https、精确到时分的官方发布时间，并且是首次发布的文本。

```powershell
& $py scripts/14_build_official_events.py --source <上游 JSONL> --validate-only
& $py scripts/14_build_official_events.py --source <上游 JSONL>
```

把输出的 `*_features` 目录填入 `market_channel.official_events`。

### 5. 启用 market_channel 并重建面板

在 `configs/llm.yaml` 中取消 `market_channel` 段的注释，然后：

```powershell
& $py main.py --new --step panels
& $py main.py --step validate
```

面板 `manifest.json` 中的 `market_quality` 会记录：新闻路由总数与保留数、证据来源分布、官方事件数、市场文章数与独立事件数、推测性摘要数，以及各切分的文章数 / 事件数 / 有效日期 / 事件类型分布。`provenance.jsonl` 中的每条市场记录都带有 `market_event_id`。

市场特征的计数项 `log1p_article_count` 改为按独立事件计数，同一事件的多篇报道合计只占一个事件的权重。

### 6. M6 门槛

`gates` 中所有切分的 `min_events` 和 `min_active_days` 必须在首次运行 E3 之前写定。值为 null，或实际覆盖低于门槛，`main.py --step validate` 都会把原因写入 `readiness.json` 并阻断正式融合。门槛不能根据测试集结果修改。

## 实验对照

以下三个版本分别与同协议的 E0 比较，不能只报告合并后的结果：

| 版本 | 配置 |
|---|---|
| 新闻路由 | `news_routes: true`，`official_events: null` |
| 官方事件 | `news_routes: false`，填写 `official_events` |
| 合并 | 两者都启用 |

`evidence_basis` 加入 `title_only` 或 `unverified_summary` 的版本只作为消融，不计入正式门槛。

## 测试

必须使用 stockmixer 环境的解释器；Windows 的 `py` 启动器会选到系统 Python，缺少 numpy 等依赖。
`tests_news` 没有 `__init__.py`，`unittest discover` 无法使用，请按模块名运行：

```powershell
& $py -m unittest tests_news.test_market_channel
Get-ChildItem tests_news\test_*.py | ForEach-Object { "tests_news." + $_.BaseName } | ForEach-Object { & $py -m unittest $_ }
```
