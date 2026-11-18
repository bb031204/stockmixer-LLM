# 本地Qwen自动审核新闻

审核模型使用本地Ollama的qwen3:8b。Qwen负责判断是否适合美国股票市场共享通道，给出证据、理由、传导机制与自评置信度；FinBERT仍负责金融情绪概率和语义向量提取，不承担本审核任务。

当前已检测到本地Ollama及qwen3:8b，模型摘要固定在configs/llm.yaml的review.expected_model_digest中。所有审核参数也在review内；原始新闻、冻结快照和三个原始待审核模板保持只读。未自动下载模型、发布路由或启用训练融合开关。

从StockMixer根目录运行，使用可导入项目依赖的stockmixer解释器：

```powershell
python -B -m news_fusion routes-auto --dataset NASDAQ --check-model
python -B -m news_fusion routes-auto --dataset NASDAQ --limit 20
```

check-model仅验证本地服务和模型版本；limit先检查一小批。也可以重复传入 --source-id 选择代表性新闻。模型配置默认temperature=0、seed=1234、think=false、num_ctx=8192，按JSON schema请求结果。证据必须能在原标题/摘要中找到，不能采用后续股价或编造股票ID。

检查小样本后，全量运行：

```powershell
python -B -m news_fusion routes-auto --dataset NASDAQ
```

三个市场可以依次运行：

```powershell
foreach ($market in @('NASDAQ','NYSE','SP500')) {
    python -B -m news_fusion routes-auto --dataset $market
    if ($LASTEXITCODE -ne 0) { break }
}
```

审核任务定义为共同的美国股票市场相关性。同一新闻版本、模型摘要、服务版本、提示词和推理参数下，三个数据集共用outputs/artifacts/news/review_cache；无需重复推理同一篇新闻，结果会分别绑定各自dataset。不要更改模板中的dataset来冒充其他数据集结果。

## 输出与断点续跑

每次生成新目录outputs/artifacts/news/routing_reviews/YYYYMMDD_HHMMSS_数据集_routes-auto/：

| 文件 | 用途 |
|---|---|
| market_review.jsonl | 完整快照的审核表，与现有routes-publish兼容；未选中和失败记录保留pending |
| audit.jsonl | 每次请求结果、原始响应、失败原因及缓存命中记录 |
| quality_sample.jsonl | include/exclude/pending各最多10条，供抽检 |
| model.json / settings.json | 实际模型版本、服务信息和推理配置 |
| prompt.txt / response_schema.json | 实际提示词和结构要求 |
| run_status.json | 进度、错误、中断状态 |
| manifest.json | 完成后生成的来源、版本和文件哈希 |

模型返回include/exclude仅在schema、证据和一致性检查通过且自评置信度达到阈值时保留。低置信度、无法判断、超出输入上下文界限或重试后仍失败的记录均为pending。默认阈值0.85是模型自评阈值，不是85%正确率保证。

脚本仅访问loopback本地服务，不使用代理或跟随重定向，不接受云托管模型。模型摘要不符或运行中模型改变会失败；输出保留已完成进度和审计。

每条成功的结构化响应都会原子写入共享缓存。Ctrl+C后重新运行相同命令即可自动复用已完成记录，生成新版本目录；不用从头推理。原输出不会覆盖。失败响应不作为有效缓存，重跑时会重试。原始模型置信度在缓存命中后仍按本次阈值重新判断；模型、提示词、文章或推理参数变化会隔离缓存。不要修改缓存制造审核结论。

自动结果明确标注review_method=local_llm，reviewed_by记录实际模型及摘要，reviewed_at使用真实推理完成时间；缓存复用保留原完成时间。它们不被标记为人工审核。human_sample_checked默认为false，建议复核所有异常样本及include/exclude边界样本；如需修改，复制market_review.jsonl为新的审核版本并保留机器原始证据。

## 发布和后续实验

脚本不自动执行routes-publish或训练。检查审核结果后，显式发布，例如：

```powershell
python -B -m news_fusion routes-publish --dataset NASDAQ --review <审核输出目录>/market_review.jsonl --allow-partial
```

保留全部行；pending不会进入市场通道，全部pending不可发布。发布后再配置market_routes.<市场>、生成真实FinBERT特征/交易日程/面板并开展融合实验。Qwen和FinBERT均在训练前离线运行，不进入forward或epoch。当前历史资料仍需标记为retrospective，真实审核和编码完成时间不能回填为过去。

接口依据：

- Ollama chat与结构化输出：https://docs.ollama.com/api/chat
- Ollama本地模型及摘要：https://docs.ollama.com/api/tags
- FinBERT模型任务：https://huggingface.co/ProsusAI/finbert