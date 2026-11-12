# 只读市场信息适配、版本核验与时间检查

本次只读消费 `D:\financial_data_project2\data\GLOBAL`、`data\US` 及必要的匹配审计/原始 ZIP，不导入采集器模块、不恢复其 outbox、不调用 Qwen/FinBERT，也不写回源项目。未改 StockMixer 模型、行情数据、根配置、训练或测试循环。

## 实际交付

新增 `news_fusion/source_adapter.py`、`adapt` 和 `validate-adapted` 命令。实际快照保存于：

`D:\finance\model\StockMixer+llm\data_llm\adapted\20261113_readonly_v1`

| 项目 | 本次快照 |
|---|---:|
| GLOBAL / US 物理 JSON | 34 / 1431 |
| source_id 去重后 | 1431 |
| 去掉的同版本视图副本 | 34 |
| 通过编码前读取检查 | 1431 |
| 必须隔离复核 | 0 |
| 当前生产审计 / 增强审计 | 256 / 1175 |
| 当前审计语言 / 原审计与不变摘要证据链恢复语言 | 256 / 1175 |
| 不同 ID 的同文本候选组 | 2，标记保留，后续按决策时点去重 |
| 纳入快照的源文件及证据文件 | 4073，哈希检查通过 |

来源仍在增量采集，因此数量大于前次分析的 1406。每次运行冻结开始时扫描到的文件清单；新增文件不混入当前版本，原清单中的内容或审计被改动、删除时则拒绝发布。校验会单独报告清单外的新文件，不为追求“最新数量”自动停止或重启采集器。

## 输出文件

- `articles.jsonl`：去重、字段适配后的摘要与元数据，供编码器的 `--input` 使用；包含标题/正式摘要，不包含原始网页全文。
- `review_records.jsonl`：需要复核的记录，本次为空；不能直接交给编码器冒充就绪输入。
- `provenance.jsonl`：每个 source_id 的全部视图路径和 SHA256、选中审计、未选中候选、语言恢复证据链。
- `time_checks.jsonl`：原始时间证据、错误与警告、有效可用时间；未知保持 null。
- `duplicate_content_groups.jsonl`：不同 ID 的相同标题/摘要哈希候选，不利用未来元数据进行全时期合并。
- `source_snapshot.json`：固定输入清单及所有读取过的源文件、审计、ZIP 的哈希。
- `manifest.json`：数量、Schema、审计/语言来源、时间统计、股票日期轴哈希、适配代码哈希与产物哈希。

“编码前就绪”只表示字段和版本适配完成，不代表已生成 FinBERT 向量、完成证券映射，或具备历史可交易 PIT 资格。

## 字段与版本规则

1. `source_id` 完整保留；同 ID 的 GLOBAL/US 副本必须内容一致，冲突时停止，不自动挑最新文件。
2. `title`、`主要内容 -> summary`、`时间 -> published_at`、`影响范围 -> impact_scope`；嵌套“时间信息”展开。缺标题不编造；公司/行业/市场层级不由目录名称推断。
3. market 优先使用内容匹配的审计 canonical_market，并核对视图关系；GLOBAL 视图与 US 地域信息不转换为 market 通道标签。
4. 当前版本审计只通过采集器原有 `record_hash` 算法关联。注意该算法的 JSON 分隔符与新闻模块旧 digest 不同；本模块明确使用 `producer_hash`，不混用。
5. 两套审计中存在多个不同候选时，仅使用与当前 JSON 内容哈希匹配者；多个匹配版本也不按时刻猜选，而是要求明确来源快照。
6. 增强旧库恢复语言时，同时验证：当前增强记录哈希、旧审计文件 SHA256、原记录哈希、原 ZIP 对应 JSON，以及 source_id/URL/来源日期/分类/关键词/摘要未改变。只继承同一摘要的语言，不把旧审计全部关系或旧可用时间套到新版本。
7. 匹配当前审计时保留模型 digest、Prompt、代码与 Schema 版本；Qwen 时间未知就保留 null，metadata-only 恢复不冒充重新运行 Qwen。
8. entities、industries、events、industry_impacts 仅保留来源已给出的内容。没有生成股票 ID、行业代码、证券历史或已验证股票链接。

## 时间规则与结果

- 来源只有日期时仍为 `YYYY-MM-DD`，不填盘前或午夜时刻。日期-only 采用严格下一交易日规则；实际日期轴可确定下一 session，但精确时刻对齐仍需要经过核验的决策日历，不猜收盘时钟。
- 保留 `source_feature_available_at`、`source_availability_basis` 和完整时间证据；canonical `feature_available_at` 保守取已知发现、抓取、Qwen 完成、提炼、验证提交及原可用时间的更晚界限，绝不回拨。
- 对读取器增加时间字段透传，observed PIT 检查也实际考虑显式 Qwen 完成、提炼、提交和有效可用时间。声明本版本需要 Qwen 但完成时间未知时，拒绝 observed PIT。
- 生产端要求可用时间严格早于 cutoff，适配后保留该规则，恰好等于 cutoff 不进入当次 observed PIT 样本。
- extraction 只有在当前内容哈希匹配且实测时间证据完整时，才规范为 `verified_audit`；来源 basis 原值及转换原因另存。migration 不升级为 observed。

本次时间告警：首次发现时间未知 3 条；需要 Qwen 但推理完成时间未知 1 条；有效可用时间需按更晚证据抬高 1 条，原值未覆盖源文件。共 850 篇有已知的当前版本 Qwen 完成时间；其余未知或仅元数据恢复，不补造推理时间。

canonical basis 为 migration 1175、verified_audit 255、extraction 1。255 仅表示上游观测证据完整，**不是完整模型级 PIT 样本数**。真实 FinBERT 完成、模型版本可用和历史证券映射仍未完成。

NASDAQ、NYSE 来源日期范围内各 699 篇；SP500 范围内 796 篇。按本批真实特征可用时间，能够在三个历史数据集结束日前就绪的文章均为 0。后续完整历史实验应明确标记 retrospective 或有明确假设的 simulated_pit，不称为真实历史 observed PIT。

## 使用

从 `D:\finance\model\StockMixer+llm\StockMixer` 执行。无需为适配安装 Transformers、PyArrow 或日历库。

```powershell
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -B -m news_fusion adapt --source-root D:/financial_data_project2 --output ../data_llm/adapted/新的版本名
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -B -m news_fusion validate-adapted --artifact ../data_llm/adapted/20261113_readonly_v1
```

适配结果目录已存在时拒绝覆盖。目录必须在生产源项目之外。原采集器若更新某个已纳入版本的 JSON/审计，第二条校验会报告变化，而不会回写或自动修补。

之后准备好本地固定权重时，可使用既有编码入口的 `--input ../data_llm/adapted/20261113_readonly_v1/articles.jsonl`。本次没有执行真实编码，没有改默认 news_root 或根训练配置，也没有开始新闻融合训练。

本次对新闻模块的改动限于新增适配器/命令，透传元数据及补齐新闻时间检查；另将产物暂存目录改为 UUID 单次创建，以避免 Windows 的 tempfile 访问拒绝被长时间重试。原模型和训练/评估约定保留。

## 验证

新增测试覆盖副本冲突、多个匹配审计、语言证据链、旧摘要变化、未知/无时区时间、晚完成 Qwen、严格 cutoff、源文件增删改、原子发布失败、清单外增量、产物不覆盖及暂存拒绝立即退出。连同原 30 项模块回归，共 49 项通过。

```powershell
& C:/Users/marti/anaconda3/envs/stockmixer/python.exe -B -m unittest discover -s tests_news -v
```

仍未在本任务内修复其他不相关的 Parquet 行号/历史映射日期类型或未来 PCA 切分兼容问题。本次产物使用 JSONL，不依赖旧的不完整 Parquet；后续使用这些路径前仍应处理相应检查项。
