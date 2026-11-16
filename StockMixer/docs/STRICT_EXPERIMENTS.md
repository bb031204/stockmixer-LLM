# 原始StockMixer对齐与两个配置

当前入口：src/train_strict.py。默认主配置：configs/stockmixer.yaml；LLM模块配置：configs/llm.yaml。

## 配置分工

stockmixer.yaml：market_name、原始lookback_length/steps/fea_num/scale_factor/activation记录、各市场stock_num/market_num/valid_index/test_index、data.protocol、训练参数、输出、自动预测及llm.enabled。

llm.yaml：只读新闻输入和适配快照、同协议baseline_checkpoint、各市场新闻panel、FinBERT离线参数、聚合/PCA/时间协议、Adapter/Gate与融合分支epochs/lr。

llm.enabled=false时不会打开LLM配置。true时必须提供真实正式E0及已审核面板，所有实际读取的配置原文与哈希保存在工作流报告。不得将有拼写错误或未接入的字段当作有效参数。

## original模式：默认

- model.py和load_data.py与官方文件逐字节一致，原get_loss和评估器保持不变。
- 原NASDAQ/NYSE四个pickle直接读取，不做额外缩放；SP500直接读取原SP500.npy并从915截断，按原公式生成收益率。
- get_batch返回原始[x,mask,price,gt]，掩码是整个输入到目标窗口的最小值。
- batch_offsets在epoch外初始化为arange(valid_index)，每轮原地打乱，取前valid_index-lookback_length-steps+1个。不是每轮重新创建一组合法目标。
- Adam lr、alpha、NumPy/PyTorch种子沿用原配置；损失分项采用原始求和/平均。
- 每轮验证及测试，只按验证loss决定best，并发布同轮test指标及预测数组；test不会用于选模。
- 打印mse、IC、RIC、prec@10、SR；RIC为Pearson IC的mean/std，SR系数15.87。

NYSE/NASDAQ：756个候选，取740个，可能使用验证目标索引756..771。SP500：1006个候选，取990个，可能使用验证目标索引1006..1021。原始尺度可能包含全样本处理信息。这些行为为复现官方代码而明确保留，不能称为无泄漏评估。

数据产物独立标记original_reproduction；所有真实采样记录在tables/epoch_samples.json。原始数据和历史结果不覆盖。

## strict模式：历史可选

将stockmixer.yaml中data.protocol改为strict，prepared_root保持null，自动使用原严格行情产物。保留合法训练目标、训练期重标定、无未来标签推理、身份/日期轴与manifest校验。不同协议checkpoint不可混用。

该选项为保留既有实验和对照而存在，不与默认original模式混名比较。历史旧schema完整配置仍可用--config显式读取。

## LLM新增模块

冻结同协议、同股票/日期轴、同预处理数据的正式E0。新闻Adapter/Gate学习收益率修正，关闭或缺失通道严格贡献0。真实FinBERT/市场路由/时间截止证据必须可审查；不伪造mask、编码或市场链接。LLM为离线预计算，不在模型forward或训练batch中调用。

当前缺少可用于正式融合的真实面板，llm.enabled默认false。仅配置好开关不代表真实新闻流水线已经运行。

## 运行与产物

直接运行src/train_strict.py即可完成检查/必要的数据准备、训练、每轮验证与测试、最佳权重预测。日常主配置默认100轮。latest指历史数据末日：NASDAQ/NYSE为2017-12-08，SP500为2022-05-25。

训练和预测分开保存到outputs，工作流报告同时保留配置原文和来源hash。价格为数据集归一化价格，不是美元。旧3个配置已归档，活动configs只保留stockmixer.yaml和llm.yaml。
