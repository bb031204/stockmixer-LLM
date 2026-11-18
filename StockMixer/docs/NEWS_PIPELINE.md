# 市场信息模块：配置、分步运行与验收

所有操作在 `D:\finance\model\StockMixer+llm\StockMixer` 中进行，使用
`C:\Users\marti\anaconda3\envs\stockmixer\python.exe`。

## 1. 保留的主干与实验边界

`src/model.py`、`src/load_data.py`、原始损失、训练与评估函数保持。
窗口16、预测步长1、5个行情特征、原始16+16+8时间分支、各市场股票数/market_num不变。
原始默认学习率0.001、alpha=0.1、100轮、两组随机种子不改。
新增新闻网络为独立 company/industry/market Adapter + Gate，FinBERT与行情主干冻结，缺失mask的修正严格为0。
FinBERT/Qwen仅在训练前运行，训练循环不调用语言模型。

**必须区分两种基线**：关闭LLM时，`stockmixer.yaml` 的 `data.protocol: original` 仍复现原代码。
原代码部分训练窗口会跨进验证标签，不能同时称为严格无泄漏。
新闻流水线的 `llm.yaml → pipeline.data_protocol: strict` 对新增实验及配对E0使用严格窗口、仅训练期拟合行情归一化；结构、损失、评价、超参数不变。
原始复现E0不能充当严格融合的父权重；程序检查协议及数据manifest，禁止混用。

严格目标索引：

| 数据集 | 训练 | 验证 | 测试 |
|---|---|---|---|
| NASDAQ、NYSE | 16～755 | 756～1007 | 1008～1244 |
| SP500 | 16～1005 | 1006～1258 | 1259～1610 |

新闻使用目标前一交易日的决策截止时间。验证/测试窗口可以包含过去的训练期行情，这属于预测时已知的历史上下文。

## 2. 两份公开配置

`configs/stockmixer.yaml`：数据集、原主干参数、alpha、E0训练轮数/学习率/种子/设备、预测日期、输出、LLM开关。

`configs/llm.yaml`：Qwen地址/固定digest/推理参数/审核阈值；FinBERT目录/长度/批量/设备；
三个通道开关与融合宽度/dropout/学习率/轮数；PCA与时间策略；预处理数据集、缓存、主数据输入、复用产物与验收阈值。
模型版本与解释器位于 `pipeline`，默认模型目录位于指定Conda环境内。
原主干的16天、1步、5特征等结构性约束保持；新闻维度可由projection_components调整，news_dim必须等于projection_components+8。

关键字段：

| 字段 | 用途 |
|---|---|
| stockmixer.llm.enabled | `train_strict.py` 是否启用融合 |
| stockmixer.market_name | 单次 `train_strict.py` 使用哪个市场 |
| llm.pipeline.datasets | `main.py` 准备/运行哪些市场，可选一个或三个 |
| llm.fusion.enabled_channels | `[company]` / `[industry]` / `[market]` / 三者组合 |
| llm.panel.mode | `retrospective`、`simulated_pit`、`observed_pit` |
| llm.pipeline.security_master | 已核验历史证券主数据JSONL |
| llm.pipeline.industry_memberships | 已核验历史行业成员JSONL |
| llm.pipeline.industry_aliases | 经核验的行业名称到标准代码词典JSONL |
| llm.pipeline.infer_industries | 有核验词典时离线运行Qwen行业推断 |
| llm.pipeline.review_artifacts | 各市场已完成的全量自动审核目录，可复用 |
| llm.pipeline.features | 同冻结快照的真实FinBERT特征目录，可复用 |
| llm.pipeline.allow_partial_routes | 只发布已通过的决定，待审核条目排除且数量明示 |
| llm.pipeline.state | 可续跑状态文件；改变配置后使用 `--new` 新建版本 |

`review_limit`、`encoding_limit` 仅用于小样本验证，正式准备阶段要求两者为null。
降低confidence阈值不能替代证据核验。模型自评confidence并非已校准的正确率。

## 3. 首次使用：逐步执行

```powershell
cd 'D:\finance\model\StockMixer+llm\StockMixer'
$pyStock = 'C:\Users\marti\anaconda3\envs\stockmixer\python.exe'

# 01 在指定环境安装依赖，保护现有torch/numpy/pandas/PyYAML，下载固定版本模型
& $pyStock scripts/01_setup_news.py

# 生成待填写映射模板；不把数组日期范围冒充股票上市/行业有效期
& $pyStock main.py --step templates

# 02 全量本地Qwen审核，再发布带证据、版本绑定的市场路由
& $pyStock scripts/02_review_news.py

# 03 冻结快照全量FinBERT编码；可复用逐文章缓存
& $pyStock scripts/03_encode_news.py

# 填写、核实模板并在llm.yaml配置主数据路径。
# 04 公司映射、行业推断、历史成员关系校验；不修改原始新闻
& $pyStock scripts/04_map_news.py

# 05 导出真实交易所日历/提前收盘，生成各市场独立PCA与三通道面板并验收
& $pyStock scripts/05_build_news_panels.py

# 06 训练相同strict协议的E0，保存权重并准备融合
& $pyStock scripts/06_prepare_news_training.py

# 最后在stockmixer.yaml设置llm.enabled: true，并选择market_name
& $pyStock src/train_strict.py
```

每一步成功后再运行下一步。审核、编码可各自复用缓存，但不要并行运行共享状态的流水线步骤。
新入口有进程锁，异常退出后操作系统释放锁。
已有完成产物不会覆写；失败后修复原因再运行。配置/历史映射变化时运行 `main.py --step templates --new` 建立新准备版本，旧产物保留。
仅调用 `train_strict.py` 不会偷偷下载模型或运行语言模型；预处理是显式步骤。

一键流程：

```powershell
# 先将stockmixer.yaml中的llm.enabled设为true，再一键执行新闻全流程
& $pyStock main.py             # setup→review→publish→encode→relations→schedule→panels→validate→baseline→prepare→train
& $pyStock main.py --step status
```

`main.py` 同样遵循公开的LLM总开关：true时按pipeline.datasets完成新闻全流程；false时只按market_name运行纯行情训练与预测，不读取LLM配置。
显式的setup/review/encode等预处理步骤可以在开关为false时提前运行。
遇到缺主数据、启用通道无训练新闻、PCA秩不足、时间或身份不一致，会在训练前停止。
单独使用 `train_strict.py` 时仍由公开的 `llm.enabled` 决定模式；两个入口均保留训练与预测在同一实验文件夹。

## 4. 新闻审核与发布

Qwen负责判断新闻能否影响**共享的美国股票市场**，不把普通单家公司事件直接广播给所有股票。
保存原始回答、逐字证据、经济传导机制、决定、模型digest、实际审核完成时间、失败与待审核状态。
`include/exclude/pending` 是市场通道决定；`exclude` 不等于该文章不能用于公司或行业通道。
已通过决定经独立校验后发布 `market_routes`，审核与发布真实时间不回填历史。
机器审核标为 `local_llm`，不冒充人工审核；`quality_sample.jsonl` 供后续人工抽检。

如果输入过长、截断、JSON不合规、引文不属于原文、证据不足或低confidence，则不自动通过。
缓存绑定文本/文章版本、模型、Prompt和推理配置。

三市场共享同一美国市场审核决定；复用时只变更dataset字段，保留原决定、证据、完成时间和完整上游审计。
待审核条目不会因切换股票池而随机重判，复用过程不标作新的模型审核调用。

行业推断另有独立缓存与逐条审计。Qwen只能选择已核验、在文章日期有效的词典条目，输出逐字证据和经济机制；
不得生成股票列表或猜ticker。没有历史行业词典时记录 `no_verified_taxonomy`，不编造通道覆盖率。

## 5. 历史证券与行业主数据

`main.py --step templates` 输出三份JSONL：`security_master.jsonl`、`industry_memberships.jsonl`、`industry_aliases.jsonl`。
证券模板覆盖三个数据集的全部数组行，默认全部pending，其余两份为空。
模板只是待审核输入，不能将pending改为verified而不补证据。

已核验证券行至少包含以下字段（这是待填写的结构示例，补齐真实证据前保持pending）：

```json
{
  "dataset": "NASDAQ", "stock_uid": "NASDAQ:AABA", "array_index": 0,
  "dataset_ticker": "AABA", "ticker": "该历史期间已核实的代码", "exchange": "NASDAQ",
  "security_id": "可靠且稳定的证券/实体标识", "company_name": "该期间正式公司名称", "aliases": [],
  "valid_from": "YYYY-MM-DD", "valid_to": null, "known_from": "实际核验完成的带时区时间",
  "mapping_source": "来源链接或证据文件", "evidence": "支持身份及有效期的证据",
  "reviewed_by": "实际审核者", "verification_status": "pending"
}
```

`valid_from` 包含，`valid_to` 排除。历史代码可不同于数据集静态代码，但必须由主数据明确关联同一稳定身份。
同一证券在不同数据集分别使用对应的stock_uid和array_index，绝不跨市场复制行号。
公司名称仅允许核验过的精确别名；代码与名称、交易所冲突时保留未匹配，不回退猜测。
仅有代码、没有公司名时，要求正文存在明确的 `NASDAQ: CODE` / `NYSE: CODE` 形式并有对应历史主数据。
数字字段不是数组位置；无效类型排除。代码复用存在歧义时不选择第一条记录。

行业成员行包含相同身份字段、`industry_standard`、`industry_version`、`industry_code`、`industry_name`、
`exposure_weight`（0～1，必须大于0；第一版成员关系权重1）以及有效期、来源、证据、审核字段。
成员有效期必须包含于对应已核验证券有效期，稳定security_id一致。
GICS不是强制唯一选择，其他可靠标准也可，但标准、版本和代码必须一致，不能混用名称拼接。

行业词典行包含 `alias`、标准/版本/代码、有效期、来源与核验字段。
单纯legacy行业标签不自动提升为核验通过的新闻行业链接。
`exposure_weight`参与文章相对聚合权重及有效文章计数；它不直接指定涨跌，也不保证同业同幅影响。

当前已有股票表中的SP500名称/sector是静态快照；NASDAQ/NYSE部分名称也是跨表静态匹配，
不能无条件推断其整个历史期间公司名、代码、行业都有效。缺少历史主数据是需要真实补齐的数据问题。

## 6. 时间、PCA和三通道验收

对每个数据集单独导出 `exchange_calendars:XNYS` 日历和实际收盘截止时间，处理夏令时、休市和提前收盘。
发布时间严格早于截止时间才可用；仅日期的新闻顺延到下一交易日，不按自然日简单填充。
在每个决策时点去重，不用未来版本全局覆盖过去信息。

当前默认24维新闻为情绪3维+训练期PCA16维+5个统计量，Adapter隐藏宽度仍为32。历史训练新闻稀疏，默认降低新闻PCA维度；这不改变原StockMixer的40维时间混合结构。
`panel.projection_components`可设1～768，`fusion.news_dim`必须为该值加8；例如设回32维PCA时news_dim=40。各市场只用该市场真实训练决策区间内已可路由新闻拟合PCA，
验证/测试只变换；不会拿SP500较晚训练区间的PCA给NASDAQ/NYSE使用。
训练新闻不足或秩不足直接报错；random_projection只能在config显式选择，不偷偷回退。
文章语义缓存可以跨数据集共享，但PCA不能跨不同训练边界共享。

输出包括三份特征与三份mask、股票/日期轴、日历、PCA、逐条来源和权重、关系/行业成员副本、manifest与分期覆盖率。
无新闻时特征全0、mask=0，区别于中性新闻。
`readiness.json` 分训练/验证/测试、三个通道分别报告覆盖率与训练阻断原因。

时间模式：

- `retrospective`：按来源发布时间回溯的历史模型重放，保留实际采集、Qwen/FinBERT完成时间，不能称为严格PIT交易回测。
- `simulated_pit`：假设发布后配置的延迟，仅是模拟，不能替代真实采集证据。
- `observed_pit`：使用发布、采集、特征、模型、Qwen/FinBERT及关系核验等实际可用时间的最大值；缺证据拒绝。

现有历史新闻在2026年加工。2013～2022年的股票区间通常没有可用的observed_pit新闻；程序不会将其改早以填充训练集。
即便发布对齐与数据切分通过，回溯实验也不能排除后训练语言模型知识、历史摘要质量带来的偏差。

## 7. 产物与恢复

`outputs/artifacts/news/pipelines/YYYYMMDD_HHMMSS_数据集_preparation/` 保存准备配置、状态、关系、日历、路由、面板与验收报告。
`outputs/artifacts/news/pipeline_state.json` 为指向当前准备版本的状态；旧目录完整保留。
审核和FinBERT缓存不在原始数据目录，不覆盖crawler JSON。

正式训练继续写 `outputs/YYYYMMDD_HHMMSS_数据集_E0或fusion/`，训练、每轮test、best权重预测都在同一文件夹。
权重按val_loss选择，不按test选模；每轮test只是沿用原代码的报告方式。
预测使用同一best权重，原五项指标及终端格式继续沿用现有实现。

配置、输入映射或源版本变更会使准备状态失效，需要新建版本重新验收。
未匹配条目保存在 `unmatched.jsonl`；不会为了使程序跑通自动填充错误行业、股票或时间。

FinBERT固定版本来源：[ProsusAI/finbert模型卡](https://huggingface.co/ProsusAI/finbert)，
当前模型卡标注CC BY-NC-SA 4.0，版本与下载文件SHA记录在Conda环境的模型目录中。


## 续跑与配置复核（本轮修复）

完成标记只表示该步骤曾执行，续跑仍会校验产物哈希、股票和日期轴、来源、时间协议、PCA维度、聚合窗口、衰减、FinBERT长度及池化。
已接受manifest的哈希固定在状态文件中；改变缓存或源数据后使用新准备版本，不能靠completed标记跳过检查。
直接指定review_artifacts时，其模型、digest、Prompt、审核阈值及冻结快照必须与当前配置匹配。
修改审核配置后可将相应review_artifacts设为null，并用--new建立新版本；逐文章有效缓存仍可复用。

FinBERT编码暂时设置CPU线程数，并在成功或失败后恢复；后续原行情训练不继承编码线程配置。
finbert.model_dir指定时也是setup的优先模型安装位置；为空时使用pipeline.model_directory。
手动指定的baseline_checkpoint或panel不会被另一项的自动解析结果覆盖。

行业推断保存event_type、impact_role、direction、confidence、importance、horizon及证据、机制和真实推断完成时间。
未知信息使用明确unknown或空列表；失败、超长输入和低置信度结果不沿用旧industry_impacts。
成功响应及缓存命中的原始回答均进入审计；失败目录保留prompt、schema、回答、错误、部分状态，且不发布完成manifest。
重试使用新目录和既有有效逐文章缓存，原失败审计不覆盖。
公司关系还会核验新闻文本与input_text_hash，证券/行业标识须为明确字符串，历史知识时间区间不可倒置。

训练前除新闻mask外还检查完整行情输入和目标价格mask；只落在无效行情窗口的新闻不算可训练。
这些检查通过不代表已完成真实行业数据、人工语义抽检、strict E0或E1--E7消融实验。

## SEC-SIC历史快照与语义复核（2026-11-17）

本轮已接入第一批真实SEC历史申报证据，使用SEC-SIC而非GICS。范围为NICK、NSP、WAT三个发行人，
并按三个数据集分别核对array_index。它是有覆盖缺口的逐日期观察表，不是全股票池连续历史主表。
未取得代码、交易所或发行人证据的申报继续pending；不按当前代码表回填，也不把两个观察日之间假定为连续有效。
证券ID为明确标注作用域的本地CIK/普通股/代码组合，不冒充商业数据库永久证券ID，也不用于合并优先股和其他股类。

原始SEC索引与证券文件均保存本地副本、URL、下载时间和SHA256。证券表、行业成员表、词典与逐条audit可以离线复核：

```powershell
& 'C:/Users/marti/anaconda3/envs/stockmixer/python.exe' scripts/07_build_historical_master.py
& 'C:/Users/marti/anaconda3/envs/stockmixer/python.exe' scripts/08_check_semantic_review.py
```

`configs/llm.yaml`的`historical_master`配置证据采集对象、缓存、版本输出目录、请求参数与offline_only。
默认离线复查现有产物；增加发行人或获取新的证据时，设置新output路径并将offline_only改为false。
采集只读取冻结快照中的raw_news.zip，校验原始记录hash，不混用后来改动的采集目录。

新增`panel.relation_time_basis: event_date`明确表示：**按照新闻发布日期已核验的证券/行业快照归属事件**，
随后仅按原来的lookback_sessions保留该事件影响。它不代表成员关系持续到下一次观察，也不允许拿某日快照匹配其他日期的新新闻。
例如1月30日的快照可以给1月30日的新闻归属，并让该事件影响在1月31日继续存在；不能据此匹配1月31日新出现的新闻。
`decision_and_event`保留原有的发布日与决策日均在成员有效期内的规则，可在config选择。
两种选择都保留真实信息可用时间、observed_pit的known_from检查、PCA训练期拟合及新闻窗口长度；不修改行情主干或训练标签。
这里event_date是published_at的纽约日期，不是摘要中提到的经济事件发生日；不能将2月12日披露的2月6日协议倒填到2月6日。

`pipeline.semantic_review`指向独立的语义抽检证据包。现有30条中，3条是用户在当前会话对所列摘要的明确认可，
27条是Codex辅助复核；仅3条计入human_confirmed_samples。其余待人工清单保存在human_review_pending.jsonl。
样本来自原审核每类前10个ID，是针对性复核，不能当作随机抽样准确率。

复核结果在新版本发布时重放，原自动审核目录与原始新闻不覆盖。辅助复核可以收紧已有市场决定，不能自动将pending/exclude提升为全市场include。
公司/行业新闻不会仅因无法匹配而自动变成市场新闻。发现发行人冲突或没有实质内容的两篇摘要，隔离全部三个通道并从PCA拟合来源中排除。
每份新市场路由和面板同时保留原决定、抽检记录及用户确认范围，加载时重新核验，不只信任human_sample_checked布尔值。

首次启用新配置时，需要新建准备版本并重建关系、路由和面板；FinBERT仍复用已验证的1423篇缓存，不重新编码：

```powershell
python main.py --step encode --new
python main.py --step review
python main.py --step publish
python main.py --step relations
python main.py --step schedule
python main.py --step panels
python main.py --step validate
```

按readiness.json判断每个市场的三个通道是否真的有训练新闻。部分主数据不保证三市场三通道全部达到门槛；
仍为0的通道继续阻止正式三通道训练，不能用降低门槛或伪造映射绕过。正式E0/融合实验仍是独立的后续步骤。

### 公司级事件不得按行业成员表扩散

真实运行中，本地Qwen提出3条行业影响：Insperity自身授信扩大、Nicholas Financial自身退市申诉、该公司的DealerCenter合作。
用户在当前会话回复“是公司”。3条均记录为行业传播exclude，仍保留公司候选，且不会改写原模型回答或伪称原自动审核已经人工通过。

`pipeline.industry_scope_review`指向这批独立人工范围核验记录；`require_industry_scope_review: true`时，新提出但尚未复核的行业传播保持pending。
`industry_scope_readiness.json`分别记录移除、待审及保留数量；作用域审核不代替证券与行业身份核验。
这批6条用户确认包括3条市场范围摘要判断和3条行业排除判断，不能扩大到全库人工审查。

`pipeline.industry_inference_artifact`可显式复用已完成的行业推断。复用必须验证新闻pin、词典hash、模型digest、Prompt/Schema和推理/阈值参数，
不以completed布尔值直接通过。更换行业词典、模型或阈值时，将该字段设null并建立新准备版本；适用的逐文章缓存仍可复用。
`pipeline.industry_keep_alive: 10m`只控制本模块的模型驻留请求，不更改全局Ollama服务或其他采集进程。

未核验不等于删除股票或强行更换通道：所有股票继续使用行情主干；已通过市场审核的新闻可提供共享市场上下文，
有公司身份就使用公司通道，有历史行业成员及有效行业事件才使用行业通道。各通道独立mask，不能保证每一篇新闻都有可用通道。

### 上游分析复用与映射审计（2026-11-17 更新）

当前默认 `pipeline.analysis_source: upstream_only`、`infer_industries: false`。
采集器负责 Qwen 语义分析，下游只做版本适配、确定性历史身份/行业/市场映射与 FinBERT 特征提取。
现有 `review_artifacts` 是有来源记录的旧审核缓存，可继续核验复用，但不称为上游新分析或人工全库审核。
未配置缓存时从上游字段生成审核产物；缺少字段的条目保留 pending，不再自动向 Ollama 请求补齐。
`industry_inference_artifact` 旧档案保留，仅在显式选择 `downstream_legacy` 且启用推断时生效；本轮流程没有模型调用。

```powershell
# 使用 stockmixer 环境，在 StockMixer 项目目录运行；只读原数据，不训练、不调用Qwen/FinBERT。
python scripts/09_check_mapping.py
```

入口读取两个 config；`pipeline.upstream_source_root` 指定采集项目根目录。
报告写入 `outputs/reports/年月日_时分秒_NASDAQ_NYSE_SP500_mapping_audit/`，包含当前冻结版本和上游候选版本的
公司/行业匹配、逐条未匹配原因、市场候选决定、完整股票轴覆盖表、原审计副本与SHA256。
候选产物是诊断材料，不能直接替代 `adapted_snapshot`、已编码特征或已批准的人工作用范围记录。
更换语料应先冻结新版本、核验原审计并重新发布路由/关系/面板；相同文章与文本/模型指纹的FinBERT逐文章缓存可复用。

字段适配保留 `industry_name_original → industry_name`、`quote → evidence`，原字段及原审计同时保留。
相互冲突的别名不选其中之一；不翻译或模糊匹配行业名称，不将 SIC 代码当作 GICS。
标准/版本/代码必须与有日期的已核验行业别名一致。明确提及的行业可依据逐字证据匹配，不补造模型置信度；
推断影响必须同时有逐字证据、机制和满足 `industry_min_confidence` 的上游自评值。
`source_confidence` 只是模型自评；确定性身份映射记录 `mapping_method/status/evidence_hash`，
`mapping_confidence` 保留 null，未经校准不伪造为100%正确率。

市场规范字段为 `market_scope: [US_EQUITY, NASDAQ_LISTED, NYSE_LISTED]` 中的适用项，
或等价的 `affected_markets` 列表；同时提供时必须一致。`impact_scope: market` 和地域标签 `US/GLOBAL`
本身不构成市场通道批准。每个范围须有 `scope_evidence` 条目：

```json
{"scope":"NASDAQ_LISTED","evidence_quote":"必须是同版本标题/摘要中的逐字引文",
 "rationale":"说明此事件为何覆盖该范围，而非仅某家公司", "confidence":0.95}
```

这是接口示例，不是真实审核记录。必须有同版本上游审计、模型revision、Prompt和真实完成时间；
只保存在原文审计但没有出现在当前编码文本的引文保持待复核，不假定摘要包含原文所有证据。
`US_EQUITY` 可作为美国股票池共享上下文；`NASDAQ_LISTED`、`NYSE_LISTED` 必须按发布日期有效的历史交易所主数据筛选。
不按数据集名称猜上市地，SP500也不能整池广播交易所限定消息。
出现交易所限定路由时市场面板保存为 `[股票, 日期, 特征]` 及相应mask；仅共享范围时沿用 `[日期, 特征]`。
读取器兼容两种布局，融合网络本身已经支持逐股票市场输入，未改主干或训练/评估函数。

对缺少连续历史主数据的股票，只保留有证据的离散观察，不自动延长有效期。
某一公司/行业匹配失败不会删除其行情样本，也不会把该新闻自动升级为全市场。
三通道的实际训练覆盖门槛仍然生效，行业覆盖为0时不允许宣称三通道实验已经准备完成。

### 上游行业字典与逐股票对应表

上游`D:/financial_data_project2/industry_data`与本项目`news_assets/industry_dictionary`共享同一587条字典及SHA。
默认使用SEC-SIC/EDGAR-SIC；TRSR基准行业和SP500原始板块独立命名空间，仅供无日期参考。
新闻分类和行业事件影响分别保存。Qwen同次正文提炼选择字典key，上游保留原文quote、范围quote、机制、自评和字典版本；FinBERT只产生文本向量，不判断股票归属。

```powershell
# stockmixer解释器；全部路径和输出版本在configs/llm.yaml的industry_dictionary配置。
python scripts/10_build_industry_mapping.py
# 等效入口，也纳入main.py的完整新闻准备流程。
python main.py --step industry_mapping --new
```

10脚本独立生成/核验对应表，不触发训练或Qwen。若相同输入产物已存在则核验复用；输入变化须配置新输出版本。
不要对同一次准备的每一步重复加--new。主干训练入口和指标逻辑保持。

输出包括：

```text
NASDAQ_stock_industry.csv / .jsonl
NYSE_stock_industry.csv / .jsonl
SP500_stock_industry.csv / .jsonl
<DATASET>_industry_to_stocks.json
stock_industry_index.jsonl
verified_security_master.jsonl
verified_industry_memberships.jsonl
manifest.json
```

逐股票表完整覆盖1026/1737/474行，每行明确stock_uid、dataset_ticker、array_index；行业→股票是多个成员的映射，不是一个行业只能有一只股票。
reference_industry_key/name仅为原数据提供的无日期参考；reference_status明确不能用于历史训练。
verified_industry_keys和观察数来自有证据的历史成员，不将未知股票猜配行业，也不扩大原SEC观察有效期。
三市场参考匹配数分别为870/1237/474，未匹配保留156/500/0行；这不代表这些股票没有真实行业，只表示当前来源未提供可靠对应。

relations步骤实际读取对应表产物中的verified_security_master和verified_industry_memberships，独立对照原历史输入和股票轴重验；参考行不能自我升级为verified。
文章标准代码须与本地字典SHA、标准/版本/代码一致，再按历史成员分配。原文引文不在摘要中时，仅在同记录hash审计、原文hash、字典版本、原始影响记录均绑定时接受原文证据；不只相信一个布尔值。
行业作用范围审核仍单独生效。字典选择不等于人工审核，不因建立字典就放行公司事件。

建表不会更新当前1423篇冻结语料或已有FinBERT特征；新采集记录须另建冻结快照和关系/面板版本。相同文本和模型的FinBERT缓存继续复用。
没有历史日期证据时仍可能保持行业覆盖0，不能用当前参考板块回填过去以绕过检查。

### 行业链路复核补强

- 字典加载器校验来源文件、命名空间及标准代码唯一性；新关系/面板同时封存字典来源文件，加载时重放SHA。旧归档按原版本读取。
- 独立建表和复用均核验原SEC证据；解析完全相同证据可进程内复用，但每次仍验证文件SHA。
- 原始参考输入按同一份字节快照计算哈希和解析，发布前复核输入未改变。
- 逐股票总表、各市场JSONL/CSV、反向索引、ticker/行号、历史核验统计和原始参考来源分别核对；自洽但与原始来源不符的分类也拒绝。
- 新行业关系使用evidence_mapping_v3，known_from取字典/推断/特征可用/人工批准时间的最晚值。已批准作用范围保留真实reviewed_at，不倒填。
- observed_pit行业训练要求v3关系；旧v2可作为归档或retrospective资料读取，不能绕过新的可用时间检查。
