# Retrace完成后如何运行

先确认上游批次全量结束、`pending=0`且`acceptance.json`中`passed=true`。程序也会自动核验这些条件和文件哈希。上游未完成时不会退回旧新闻训练。

## 先检查两个配置

只编辑 `configs/stockmixer.yaml` 和 `configs/llm.yaml`：

| 文件 | 主要配置 |
|---|---|
| stockmixer.yaml | `market_name`、`llm.enabled`、原始主干参数、E0轮数/学习率/alpha、设备、`prediction` |
| llm.yaml | `pipeline.retrace_artifact`、`pipeline.datasets`、FinBERT、三通道开关、映射、时间协议、覆盖门槛、融合训练参数 |

当前目标是NASDAQ，`llm.enabled: true`，启用`market + industry`，采用`original + retrospective`。没有自动打开公司通道或降低覆盖门槛。历史行业参考表不等同于真实历史成员证明。

切换单个数据集时，同时修改`stockmixer.yaml`的`market_name`和`llm.yaml`的`pipeline.datasets`。批量运行可将后者设为`[NASDAQ, NYSE, SP500]`，`market_name`须包含在列表中。

`pipeline.baseline_checkpoints`分别配置三个数据集的可复用E0。当前仅NASDAQ填写已有权重；NYSE、SP500为`null`，新闻验收通过后自动按主配置训练对应E0。权重市场、数据协议、数据哈希或训练参数不匹配会报错；需要重训该市场时将其设为`null`。顶层`baseline_checkpoint`保持`null`，由准备状态提供各市场的权重。

## 一键完成

在PowerShell执行：

```powershell
conda activate stockmixer
cd 'D:\finance\model\StockMixer+llm\StockMixer'
python main.py
```

解释器应为`C:\Users\marti\anaconda3\envs\stockmixer\python.exe`。程序自动执行：

1. 核验并导入全量retrace；切换新快照，清除旧快照绑定的面板与审核缓存路径，保留内容缓存和来源绑定抽检。
2. 检查FinBERT环境，离线编码，建立公司/行业/市场关系，按交易日生成面板。
3. 验收编码、事件去重、证据、价格窗口、训练期PCA和市场覆盖门槛。
4. 核验复用或训练同口径E0，再训练融合分支、逐轮验证/测试、按验证loss选模；`prediction.enabled: true`时自动用最佳权重预测。

验证失败会停止并保留原因，不会跳过缺口进入训练。`main.py`会复核并复用已完成阶段；中途失败后修正原因，再运行同一命令。已经完成的整次实验不会因重复运行而再次训练。

**想重新做一轮准备和训练：**`python main.py --new`。它创建新准备版本，保留旧实验、完整新闻快照及FinBERT内容缓存。复用E0仍须通过参数校验。中断的训练不会恢复优化器状态，重试会重新训练并保留旧目录。

## 分阶段运行

| 什么时候 | 命令 | 做什么 |
|---|---|---|
| 只查看进度、状态 | `python main.py --step status` | 只读查看上游和下游状态，不导入或训练 |
| 上游完成后先核验 | `python main.py --step check` | 检查全量验收、源文件哈希、配置及行情轴，不编码或训练 |
| 只导入新新闻 | `python scripts/18_import_news.py` | 导入并激活已验收全量批次，不训练 |
| 准备新闻与同口径E0 | `python scripts/19_prepare_news.py` | 编码、映射、面板、最终验收；**缺少可复用E0时会训练E0**；不训练融合分支 |
| 准备完成后训练、测试、预测 | `python scripts/20_train_test.py` | 使用已准备数据训练融合分支，按配置测试并预测 |

通常选“一键”或“18→19→20”任一方式即可。默认自动导入，19也会核验并补做18；20要求19已完成。

若只想做到新闻验收、暂不训练E0，可在18后按以下顺序运行低层阶段：

```powershell
python main.py --step setup
python main.py --step review
python main.py --step publish
python main.py --step encode
python main.py --step industry_mapping
python main.py --step relations
python main.py --step schedule
python main.py --step panels
python main.py --step validate
```

这里`review`复用上游Qwen结果，当前`analysis_source: upstream_only`，不会重新运行Qwen。FinBERT只在离线编码时运行。

19完成后也可以直接运行`python src/train_strict.py`；它按`stockmixer.yaml.market_name`创建一次新训练与预测，不负责导入新闻。文件名`strict`不决定实验协议；协议由config决定。原始复现协议保留验证标签重叠，不能称为无泄漏PIT。

## 输出在哪里

- `outputs/artifacts/news/launch_state.json`：最新全流程状态指针；失败阶段和原因也记录在这里。
- `outputs/artifacts/news/launches/YYYYMMDD_HHMMSS_数据集_launch/`：`launcher.log`和`launcher.json`。
- `outputs/artifacts/news/adapted/`：已导入的只读派生快照、待检查记录及证据。
- `outputs/artifacts/news/pipelines/`：FinBERT特征、关系、三通道面板、`state.json`和`readiness.json`；最终有效输入统计位于`readiness.json`的`final_inputs`。
- `outputs/YYYYMMDD_HHMMSS_数据集_E0或fusion/`：每次训练、测试与自动预测在同一实验文件夹；预测表为`predictions/predictions.csv`，测试/训练指标在`tables/`。

新批次导入前的配置备份保存在`outputs/artifacts/maintenance/`。程序不修改上游JSON，不自动启动采集或retrace。

若只跑纯StockMixer，将`llm.enabled: false`后运行`main.py`；不会读取LLM配置。`latest`预测的是历史数据末日的下一期，不是实时行情。
