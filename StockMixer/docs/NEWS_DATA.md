# 新闻输入与离线处理

原始输入是项目上一级data_llm/US、GLOBAL，只读保留。冻结适配产物位于outputs/artifacts/news/adapted；当前选定版本在llm.yaml的adapted_snapshot中配置，数量及证据以该版本manifest为准。

配置分工：stockmixer.yaml的llm.enabled控制是否融合；llm.yaml设置离线FinBERT、本地模型路径、面板、正式E0及融合分支。关闭开关时纯行情训练不读取LLM配置。当前缺少已配置的真实面板，不应把接口已实现当作正式新闻实验已完成。

从StockMixer根目录验证已有冻结快照：

```powershell
python -B -m news_fusion.snapshot validate --artifact outputs/artifacts/news/adapted/20261113_current_v1 --current-root ../data_llm
```

离线命令仍由python -m news_fusion提供，包括adapt、validate-adapted、encode、schedule、build、validate。编码前在configs/llm.yaml配置真实本地FinBERT路径；build使用已经审核的关系及真实交易日程，不能补造市场链接。源适配、去重、语言、时间证据链和mask检查均在现有news_fusion模块中，未因文档整理而移除。

Qwen/FinBERT不在训练forward或epoch中运行；未知时间、公司、行业映射不得编造。无新闻不等于Neutral，缺失通道修正为0。默认retrospective，不能宣称历史observed PIT。

旧实现过程与1431父快照、1423选定快照的历史说明保存在outputs/archive/process_cleanup_20261116/historical_docs；完整可复现证据、权重、配置快照和manifest仍保留在outputs中。
