# FinBERT 与 StockMixer 融合模块：本次落地与使用

本模块位于 `D:\finance\model\StockMixer+llm\StockMixer\news_fusion`。它提供离线编码、时间过滤、历史映射、三通道面板、窗口读取和独立包装模型。**没有接入或替换现有训练入口**，也没有新增另一套训练/测试循环。

## 1. 已核验的基线与数据

目标仓库为 `D:\finance\model\StockMixer+llm`，分支 `main`。当前基线是发布代码的 40 维路径，`scale_factor` 未实际控制其结构；它不同于另一个目录 `D:\finance\model\StockMixer` 中后来修改的 28 维版本。本次没有将两个目录的模型混用。

以下文件按修改前后 SHA256 保持不变：`src/model.py`、`src/train.py`、`src/load_data.py`、`src/evaluator.py`、`config.yaml`。没有写入已有行情文件、新闻源文件、checkpoint 或历史实验结果。

| 数据集 | 行情 shape | 核验后的日期范围 | 名义训练/验证/测试天数 |
|---|---|---|---|
| NASDAQ | `(1026,1245,5)` | 2013-01-02 至 2017-12-08 | 756 / 252 / 237 |
| NYSE | `(1737,1245,5)` | 2013-01-02 至 2017-12-08 | 756 / 252 / 237 |
| SP500 | `(474,1611,5)`，已按基线截取 | 2016-01-04 至 2022-05-25 | 1006 / 253 / 352 |

NASDAQ/NYSE 名义边界：训练截至 2015-12-31、验证 2016-01-04 至 2016-12-30、测试 2017-01-03 至 2017-12-08。SP500 名义边界：训练截至 2019-12-31、验证 2020-01-02 至 2020-12-31、测试 2021-01-04 至 2022-05-25。

SP500 股票顺序及全部规范化特征已与原始字典逐股票核对；NASDAQ 数据哈希与既有逐行核验报告一致；NYSE 对照已有日期、ticker 元数据。三个独立的轴快照位于 `news_assets/axes`，包含股票顺序、完整日期、来源哈希与参考边界。它们不是历史证券/行业主数据，也不会覆盖根配置。

本次盘点时 `data_llm/US`、`data_llm/GLOBAL` 均为空；没有真实英文摘要、历史证券映射、历史行业映射或本地 FinBERT 权重。原环境具有 Python 3.9.25、PyTorch 2.7.1+cu128、CUDA，但没有 Transformers、PyArrow、交易日历库。完整盘点见 [news_inventory.json](news_inventory.json)。

## 2. 对实施文档的取舍与修正

1. **保留用户指定的训练/测试行为。** 文档建议严格训练标签边界和最终只测试一次，但本次明确要求不改训练测试逻辑，因此均未实施。现有采样可能使用验证期前 16 个目标日，现有训练每轮计算测试表现；报告结果时仍须披露，不能把它称为已修复的严格独立评估。新增 PCA 仅使用名义训练目标能够访问的新闻，不利用上述重叠扩大拟合集。
2. **不照论文表格补天数。** NASDAQ/NYSE 的实际测试段为 237 天；不按表中的 273 天伪造日期。
3. **保持价格接口并保证零残差一致。** 使用 `prediction = baseline_price + base_price * sum(news_return_delta)`，与文档的收益率融合公式等价。零残差时避免价格/收益率往返舍入，测试验证输出与同一基线逐元素一致。
4. **不用论文发表日冒充具体权重版本可用日。** 本地模型文件用 SHA256 固定；`model_available_at` 默认未知。严格 observed PIT 需要有证据的具体模型版本可用时间。FinBERT 在今天可下载不代表它在 2013 年已存在。
5. **PCA 不引入不必要依赖。** 使用 NumPy SVD 实现 32 维 PCA，保存均值和分量，不需额外 scikit-learn/joblib。少于 33 篇有效训练文章或秩不足时明确停止；只有显式选择 `random_projection` 才使用固定种子的无拟合替代，不静默切换。
6. **处理 Windows 与缺失值。** 发布面板前释放写入映射；无新闻的 NaN 特征在进入 Adapter 前用 `where` 屏蔽，避免 `NaN*0` 污染基线。
7. **新闻年龄不因处理延迟重置。** 年龄从发布时间对应的首个可用交易时点计算，推理晚完成只推迟准入，不让陈旧新闻重新变“新”。本版本保守排除数据集首日之前的文章，避免缺乏前置交易日轴时把多年前新闻压到索引 0；因此头几天覆盖率可能偏低。

## 3. 新增模块

| 文件 | 职责 |
|---|---|
| `news_fusion/inputs.py` | 正式 JSON/JSONL/Parquet 摘要输入，文本去重、语言与版本冲突检查 |
| `news_fusion/encoder.py` | 本地冻结 FinBERT、标签重排、最后层 CLS、版本缓存、OOM 缩批 |
| `news_fusion/time_axis.py` | 数据集轴、显式决策日历、三种时间模式、日期-only 与时区处理 |
| `news_fusion/panel.py` | 已验证关系、历史行业成员、训练文章筛选、32 维压缩、40 维面板和来源账本 |
| `news_fusion/dataset.py` | 校验文件/轴哈希、mmap 读取、按输入窗口最后一天取得新闻 |
| `news_fusion/fusion.py` | 可选 `StockMixerWithNews`，三套独立 Adapter/Gate/Head |
| `news_fusion/io.py` | 文件哈希、JSONL/Parquet 和不覆盖已有版本的产物发布 |
| `news_fusion/__main__.py` | `inspect / encode / schedule / build / validate` 命令 |
| `configs/news_pipeline.yaml` | 独立新闻配置，不替代行情配置 |
| `tests_news` | 合成数据行为检查及真实行情接口回归 |

## 4. 输入接口

文章必须有非空 `source_id`，提供英文 `summary`（或 `主要内容`），可选 `title`（或 `标题`）。仅标题而无摘要会跳过。语言由可靠上游明确提供 `language=en`；未知语言、中文不伪装为有效英文特征。输出只保留文本哈希、元数据、概率与向量，不复制原文或保存标题/摘要输入。

市场归属必须由 `market=US/GLOBAL` 明确指定。`impact_scope` 为 `company/industry/market` 列表；目录名为 US 不等于市场级新闻。基础时间字段包括 `published_at`、`publication_precision`、`first_seen_at`、`fetched_at`、`feature_available_at`、`availability_basis`。`时间` 仅作为明确的发布时间字段别名；无法确认的嵌套“时间信息”不会自动猜测其含义。

可选保留 `input_schema_version`（来自原 `schema_version`）、`qwen_model_revision`、`prompt_version`、`record_version`。同一个 source_id 出现不同正文摘要或元数据版本时停止，要求先提供明确的 as-of 快照；不选择“最新版本”回填历史。

关系文件是独立 JSONL 或 Parquet，不能直接把 LLM 的 ticker 猜测当成已验证映射：

- 公司关系：`source_id, channel=company, dataset, stock_uid, valid_from, valid_to, known_from, known_to, verification_status=verified, mapping_source`。
- 文章行业关系：`source_id, channel=industry, industry_standard, industry_version, industry_code, evidence_type=explicit/inferred`，以及同样的有效期、获知期、核验状态和来源。
- 历史行业成员文件：`dataset, stock_uid, industry_standard, industry_version, industry_code`，以及上述有效期、获知期、核验状态和来源。

`valid_to`、`known_to` 是右开边界，可为空；`valid_from` 必须有真实来源。observed PIT 还必须有带时区的 `known_from`。`stock_uid` 是本数据集的行键（如 `NYSE:A`），不是永不变化的全球证券 ID。ticker 更名、复用或多类股份需要在上游证券主数据中核验，再写入相应关系。

没有历史行业成员时，行业通道保持缺失；不会用当前 sector_snapshot 回填。行业体系、版本、代码必须同时匹配。importance 只有在提供有限的 `[0,1]` 数值且 `importance_verified=true` 时才使用；否则重要性值为 0、可用比例为 0。

## 5. 时间、压缩与面板约定

默认 `mode=retrospective`。它研究历史文本与收益的关系，不宣称历史可交易 PIT。`simulated_pit` 使用明确的来源发布时间加模拟延迟，是假设回放，也不代表模型、摘要和行业映射真实存在于历史。

`observed_pit` 要求 `availability_basis=observed/verified_audit`，并取发布时间、首次发现、抓取、上游特征生成、FinBERT 完成、新闻特征可用、具体模型可用时间的最大值。证券/行业关系在决策日及原新闻日均须有效，并满足其 known 时间。缺时间、缺证据或晚于截止的记录不能提前使用。

所有决策时刻必须显式提供。`schedule` 命令使用 exchange-calendars 的 XNYS 收盘时刻，包含夏令时、节假日和提前收盘；也可提供来源可核验且与数据集日期完全相同的 `date,decision_cutoff_at` JSONL。没有日历时明确停止，不用固定 16:00 代替。

每篇文章只在发布时间对应时点后的最近 5 个交易日内参与，权重为 `exp(-age/3)`。同一通道、股票和决策日按内容哈希去重；同文的多个行业路径不会重复累计，explicit 优先于 inferred。

PCA 是各数据集独立的训练期变换。当前模块仅支持 `steps=1`，最后一个名义训练目标为 `train_end_idx-1`，因此可用于拟合的决策索引必须 `< train_end_idx-1`。还要求新闻已经能通过当前模式及映射进入该数据集。验证/测试向量只做变换。PCA 使用训练期统计的普通 holdout 协议，不代表训练期每个历史日期都已拥有最终拟合好的 PCA。

公司/行业面板为 `[N,T,40]`，市场为 `[T,40]`；mask 分别为 `[N,T]`、`[N,T]`、`[T]`。40 维按顺序为：3 个情绪概率、32 个压缩语义维度、`log1p_article_count`、`mean_age_sessions`、`inferred_share`、`mean_importance`、`importance_available_ratio`。缺新闻全部填 0、mask=0，独立于行情有效性 mask。

每个版本保存面板、mask、完整轴、决策日历、压缩矩阵、`provenance.jsonl` 和 manifest，包含来源文章、关系、模型、代码、股票/日期顺序与文件哈希。输出目录已存在则停止；不覆盖旧产物。

## 6. 运行方式

以下命令从 `D:\finance\model\StockMixer+llm\StockMixer` 执行。先检查，无需 NLP 新依赖：

```powershell
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -B -m news_fusion inspect
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -B -m unittest discover -s tests_news -v
```

原环境未被安装或升级新包。需要真实 FinBERT 和日历时，可建立独立虚拟环境复用已安装的 PyTorch：

```powershell
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -m venv --system-site-packages .venv-news
& ./.venv-news/Scripts/python.exe -m pip install -r requirements-news.txt
```

requirements 中固定独立环境的 NumPy/Pandas，避免日历依赖把原环境的 NumPy 单独升级后造成二进制不兼容。本次未执行依赖安装；该新环境仍需按本机安装结果验证。

模型下载仅在离线编码前做一次，不进入训练。以下为 2026-11-12 核对的固定 commit 下载示例；**本次未执行下载**：

```powershell
& ./.venv-news/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ProsusAI/finbert', revision='0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031', local_dir='../models/finbert', allow_patterns=['*.json','vocab.txt','model.safetensors','README.md'])"
```

应核对该 revision 的模型卡与许可证。官方模型卡标注英文、768 hidden size、三分类及 CC BY-NC-SA 4.0；代码从实际 `id2label` 读取顺序，保存统一 positive/negative/neutral 顺序。编码器只读本地目录，记录所有本地权重/tokenizer 文件哈希，不使用浮动 main，也不自动联网。

有真实正式摘要后，先编码小样本：

```powershell
& ./.venv-news/Scripts/python.exe -B -m news_fusion encode --model-dir ../models/finbert --output ../data_llm/article_features/v001 --cache ../data_llm/finbert_cache --limit 16
```

默认从 `US`、`GLOBAL` 读取正式 JSON/JSONL/Parquet；可用 `--input 文件1 文件2` 明确指定输入。默认输出 JSONL，不依赖 PyArrow；加 `--format parquet` 生成 Parquet 表。文章向量始终单独保存为 float16 NPY。缓存键覆盖 source_id、文本哈希、权重、tokenizer、池化、截断和预处理版本；缓存命中保留最初推理完成时间。

生成数据集日历，准备经过核验的关系文件，再构建并校验面板：

```powershell
& ./.venv-news/Scripts/python.exe -B -m news_fusion schedule --dataset NYSE --output ../data_llm/schedules/NYSE.jsonl
& ./.venv-news/Scripts/python.exe -B -m news_fusion build --dataset NYSE --features ../data_llm/article_features/v001 --schedule ../data_llm/schedules/NYSE.jsonl --links ../data_llm/mappings/article_links.jsonl --memberships ../data_llm/mappings/industry_history.jsonl --output ../data_llm/panels/NYSE/v001
& ./.venv-news/Scripts/python.exe -B -m news_fusion validate --panel ../data_llm/panels/NYSE/v001
```

NASDAQ、SP500 分别生成自己的日历与面板。没有公司/行业映射时可省略相应参数，仅有内容依据明确的市场级新闻能够进入市场通道。没有有效文章时停止，不制造“已完成”空面板。PCA 训练文章不足时可以显式加 `--projection random_projection` 做单独实验，manifest 会明确区分。

## 7. 后续接入接口（本次不接训练循环）

```python
from news_fusion.dataset import NewsPanel
from news_fusion.fusion import StockMixerWithNews

# baseline 是按原配置构建并已加载原 checkpoint 的 StockMixer 对象。
fusion = StockMixerWithNews(baseline, freeze_backbone=True)
panels = NewsPanel(panel_directory)
panels.assert_compatible(eod_data.shape, verified_stock_uids, verified_dates)
news = panels.window(offset, lookback=16, steps=1)
# news['decision_idx'] == offset+15；target_idx == offset+16。
# 把六个 news/mask 数组转换成与 stock_x 同设备的 Tensor 后传入 fusion。
prediction_price = fusion(stock_x, base_price, **news_tensors)
# prediction_price 可直接交给原有 get_loss，不改标签或损失。
```

`news_tensors` 仅包含 company/industry/market 的六个张量键，不包含两个索引。市场向量可以是 `[40]`，由包装器广播；公司和行业必须为 `[N,40]`，防止误广播股票轴。

三个通道独立 LayerNorm/MLP Adapter、Gate、残差头；Gate 以新闻和原基线预测收益率为条件，使共享市场新闻也能有按股票变化的门控。残差头权重和 bias 全零初始化，Gate bias=-2。冻结基线时 `train()` 会保持其 eval 状态，只有新闻分支参数可训练。默认新增可训练参数 4,377 个。

通道开关通过 `enabled_channels` 显式选择。无新闻时输出逐元素等于同一个基线；不会因新闻缺失删除行情样本。NaN/Inf 的活跃新闻或非法基准价格会报错，而不是静默修复。

**当前原 `src/train.py` 仍只训练原 StockMixer。** 运行它不会启用新闻，也不会自动读取上述独立新闻配置。本次没有运行 E0-E9 或三种子完整训练，不声称融合带来了指标提升。

## 8. 验证结果与未完成的真实输入

30 项 unittest 检查涵盖缓存失效/无明文、CLS/标签顺序/冻结契约、日期-only/周末/夏令时/提前收盘、真实与模拟可用时间、双时间映射、训练期 PCA 的实际文章筛选、陈旧新闻、三通道独立 mask、股票轴/日期轴哈希、零初始化等价、冻结与梯度、checkpoint 再加载及 CUDA 前向/反向。

真实 NASDAQ/NYSE/SP500 行情批次已经过不改动基线的前向和原损失检查，零新闻输出逐元素一致。FinBERT 后端的单元检查使用明确标记的合成模型/向量；**尚未用真实 ProsusAI 权重跑推理**。测试结果不能当作真实新闻编码或投资效果验证。

真实运行还需要：正式英文摘要及稳定 source_id、可核验时间字段、本地固定版本权重、公司/历史行业关系。observed PIT 另需具体模型版本及关系版本的历史可用证据。固定股票池仍不能宣称为逐日真实指数成分，原 RIC 仍是 mean(IC)/std(IC)，均未随本模块改动。

参考： [FinBERT 模型卡](https://huggingface.co/ProsusAI/finbert)、[固定版本模型 API](https://huggingface.co/api/models/ProsusAI/finbert)、[Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。
