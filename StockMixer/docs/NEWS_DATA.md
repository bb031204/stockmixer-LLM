# 新闻输入与离线处理

原始输入位于项目上一级 data_llm/US、GLOBAL，只读保留。默认编码读取 configs/llm.yaml 中 adapted_snapshot 指向的冻结快照，先验证完整哈希、来源和成员证据，再读取 articles.jsonl；不再自动扫描原始 JSON，不会猜测语言。

无需本地模型即可检查编码输入：

```powershell
python -B -m news_fusion encode --check-input
```

当前1423篇冻结新闻均为 ready。`--input` 可显式覆盖默认快照，所有相对输入路径均相对于 StockMixer 项目根目录；错误输入会失败，不回退扫描其他目录。默认快照验证针对冻结证据自身，不要求活跃采集目录永远保持旧版本。若需另行核对当前原始文件，可运行：

```powershell
python -B -m news_fusion.snapshot validate --artifact outputs/artifacts/news/adapted/20261113_current_v1 --current-root ../data_llm
```

## FinBERT编码

按当前要求统一使用 `C:/Users/marti/anaconda3/envs/stockmixer/python.exe`。先运行 `scripts/01_setup_news.py`，依赖和固定版本本地ProsusAI/finbert均位于该Conda环境，原Torch/NumPy/Pandas/PyYAML版本受到保护。完整分步与一键入口见[NEWS_PIPELINE.md](NEWS_PIPELINE.md)。也可单独执行：

```powershell
python -B -m news_fusion encode
```

输出包含情绪概率、768维向量、模型及分词器版本、实际完成时间、输入快照和输入文件哈希。FinBERT保持冻结，训练中只读取已生成的特征。`--check-input`不加载模型、不代表已完成真实编码。`main.py --step setup`显式准备依赖并下载固定版本模型；已安装且哈希一致时可离线复用。编码与训练阶段不在线下载模型。实际完成数量和模型哈希以编码manifest及pipeline_state.json为准。

## 市场路由审核与发布

impact_scope、US/GLOBAL目录或市场标签均不代表已完成市场相关性审核。新流程为：

```text
已验证冻结快照 -> routes-template待审核表 -> 审核者填写新版本JSONL
  -> routes-publish校验发布 -> build读取审核产物
  -> 面板携带审核证据 -> 训练加载再次验证
```

生成目标数据集的待审核表：

```powershell
python -B -m news_fusion routes-template --dataset SP500
```

输出位于 outputs/artifacts/news/routing_reviews/时间戳_SP500_routes-template/market_review.jsonl。所有行默认pending，即使上游已有market标签也不自动通过。表内包含source_id、新闻版本pin、标题、摘要、发布日期和候选scope，便于审核。

请将market_review.jsonl复制为一个新的reviewed.jsonl再修改，保留原模板及其manifest。逐条填写：

| 字段 | 要求 |
|---|---|
| decision | include / exclude；暂未判断保持pending |
| evidence_quote | include时必须逐字引用该新闻标题或摘要中的证据，至少8个字符 |
| rationale | include/exclude均需说明市场相关性判断理由，至少8个字符 |
| reviewed_by | 实际审核者标识 |
| reviewed_at | 实际审核完成时间，ISO8601且包含时区；不得填写未来时间 |

不要修改dataset、source_id、article_pin、标题、摘要和原始发布时间。审核关注是否确实影响美国股票市场，不能只因文件放在US目录就include。程序验证身份、版本、证据出处及记录完整性；语义判断仍需有依据地完成，不能批量伪造审核结论。

发布已审核版本：

```powershell
python -B -m news_fusion routes-publish --dataset SP500 --review outputs/artifacts/news/routing_reviews/你的审核版本.jsonl
```

默认要求所有行均已include/exclude；仍有pending会拒绝发布。若只审核了一部分，必须明确加 `--allow-partial`，保留全部行，pending记录会计数但不会进入市场通道。全是pending的表在任何模式下均不可发布。

审核产物保存绑定的新闻版本、原审核记录、已批准路由及文件哈希。known_from取实际发布/审核完成时间的较晚者，不能通过回填较早的reviewed_at制造历史PIT证据。复制或编辑布尔标记不能替代这些证据。

## 面板生成与训练

在 llm.yaml 的 market_routes.SP500 填已发布的路由目录，或在build命令传 `--market-routes`。离线产物示例：

```powershell
python -B -m news_fusion schedule --dataset SP500
python -B -m news_fusion build --dataset SP500 --features <真实编码产物目录> --schedule <交易日程JSONL> --market-routes <已发布审核目录>
python -B -m news_fusion validate --panel <面板目录>
```

build核对审核与编码的快照/文章版本，仅将已批准且时间有效的路由用于market通道。路由证据被复制进新面板并记录manifest哈希；训练读取时再次核对数据集、证据、批准清单和市场provenance。没有真实FinBERT产物、有效日程或审核路由时，不应生成可训练的市场面板。

旧版程序接口允许生成未审核的诊断面板，但market_routing_verified为false，不能进入市场融合训练。直接把false改true会因缺少审核证据而失败。CLI启用market通道时会直接要求审核产物。

公司/行业关系仍须通过已核验的股票身份和历史行业关系表提供，不能由市场审核代替。三个通道分别保留mask，无新闻不等于Neutral，缺失修正为0。默认retrospective；2026年采集或编码的历史文章不能冒充2013—2022年observed PIT可用特征。

训练前需完成同strict协议正式E0及面板验收；baseline_checkpoint和panels.<市场>可手动填写，也可保持null从已验收的准备状态读取，然后启用stockmixer.yaml的llm.enabled。train_strict.py仍只负责训练、评估和预测，不会在epoch里运行Qwen或FinBERT。

## 本地大模型辅助审核

已增加python -m news_fusion routes-auto，使用llm.yaml的review配置调用本地qwen3:8b，生成带模型来源标识、证据和审计的审核文件；低置信度或失败记录保留pending。完整操作及缓存续跑说明见[AUTO_REVIEW.md](AUTO_REVIEW.md)。FinBERT仍用于情绪与向量，审核不进入训练循环。
