<div align="center">
  <h1>Prompt Optimizer</h1>
  <p>基于标注数据，自动优化 LLM Prompt 的轻量级工具</p>
</div>

-----------------

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/prompt-optimizer.svg)](https://github.com/shibing624/prompt-optimizer/issues)

**Prompt Optimizer** 通过 Worker-Master 双模型架构，迭代式地分析分类错误、改写 Prompt，逐步提升 Prompt 在目标任务上的准确率。无需手动调 Prompt，无需训练模型。

## 工作原理

```
                    ┌─────────────────┐
                    │   标注数据 CSV    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Train/Val 拆分  │  固定种子，防过拟合
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌─────────────────┐           ┌─────────────────┐
     │  Worker 预测     │           │  Worker 预测     │
     │  (Train 集)      │           │  (Val 集)        │
     └────────┬────────┘           └────────┬────────┘
              │                              │
              ▼                              ▼
     ┌─────────────────┐           ┌─────────────────┐
     │  收集错误样本     │           │  计算 Val Score  │
     └────────┬────────┘           └────────┬────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Master 错误分析    │  分析失败原因，提炼通用规律
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │  Master 增量改写    │  小幅修改 Prompt，保留有效规则
                  │  Prompt            │
                  └─────────┬──────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Val Score 提升？  │
                   └────────┬─────────┘
                     是 /        \ 否
                       ▼          ▼
                  保留新 Prompt   回滚到 Best Prompt
                       │          │
                       └────┬─────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Git Commit + results.tsv │  记录每轮结果
              └──────────┬───────────────┘
                         │
                         ▼
                  ┌──────────────────┐
                  │ 继续迭代 /       │
                  │ Early Stop /     │
                  │ 达到最大轮数      │
                  └──────────────────┘
```

**关键设计：**
- **Train/Val 拆分** — 固定种子拆分，防止 Prompt 过拟合到训练数据
- **增量改写** — Master 只做小幅修改，保留已有效的规则，避免推倒重来
- **多指标评估** — 同时计算 Accuracy、F1、Precision、Recall，支持选择主指标
- **多数投票** — 同一 query 可跑多次取多数票，减少 LLM 输出随机性
- **自动回滚** — val score 下降时回滚到最优 Prompt，在 best 基础上继续优化
- **Git 集成** — 每轮自动 commit prompt 文件，方便回溯任何历史版本
- **结构化记录** — `results.tsv` 记录每轮指标，可程序化分析
- **Early Stop** — 连续 N 轮无提升则自动停止，节省 API 调用
- **异常重试** — LLM 调用失败自动指数退避重试，避免网络抖动影响实验

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key 和模型配置：

```env
# Worker 模型（执行分类任务，推荐用快速且便宜的模型）
WORKER_API_KEY=your_worker_api_key
WORKER_BASE_URL=https://api.openai.com/v1
WORKER_MODEL_NAME=gpt-4o-mini

# Master 模型（错误分析 + prompt 改写，推荐用更强的模型）
MASTER_API_KEY=your_master_api_key
MASTER_BASE_URL=https://api.openai.com/v1
MASTER_MODEL_NAME=gpt-4o
```

支持任何兼容 OpenAI API 格式的模型服务（OpenAI、Azure、本地部署等）。

### 2. 准备数据

准备 CSV 格式的标注数据文件 `data.csv`：

| question | 是否事实 |
|---|---|
| 怎么投私信获取线索？ | 事实 |
| 为什么我的流量挣钱越来越少？ | 非事实 |

列名和标签值可在 `prepare.py` 中通过 `TEXT_COLUMN`、`LABEL_COLUMN`、`LABEL_MAP` 自由配置，适配任意分类任务。

### 3. 编写初始 Prompt

在 `prompt.md` 中编写初始 Prompt，使用 `{{query}}` 作为用户输入的占位符。

### 4. 运行优化

```bash
python run.py
```

支持命令行参数覆盖默认配置：

```bash
python run.py --iterations 10 --patience 5 --metric f1 --vote-count 3
```

优化过程中会：
- 在终端实时输出每轮的多指标分数（Accuracy/F1/Precision/Recall）
- 自动将最优 Prompt 保存到 `prompt.md`
- 每轮自动 Git commit，记录 prompt 版本历史
- 结构化结果记录到 `results.tsv`
- 完整实验历史记录到 `research_log.md`

## 配置说明

在 `prepare.py` 中可调整以下参数，也可通过命令行覆盖：

| 参数 | 默认值 | CLI 参数 | 说明 |
|---|---|---|---|
| `ITERATIONS` | 5 | `--iterations` | 优化迭代轮数 |
| `SAMPLE_SIZE` | 200 | `--sample-size` | 每轮从 train 集抽样评估的条数 |
| `CONCURRENCY` | 10 | `--concurrency` | LLM 请求并发数 |
| `VAL_RATIO` | 0.2 | `--val-ratio` | 验证集比例 |
| `PATIENCE` | 3 | `--patience` | 连续 N 轮无提升则 early stop |
| `VOTE_COUNT` | 1 | `--vote-count` | 多数投票次数（1=不投票，3/5=投票） |
| `PRIMARY_METRIC` | accuracy | `--metric` | 主评估指标（accuracy/f1/precision/recall） |
| `MAX_RETRIES` | 3 | - | LLM 调用失败时的最大重试次数 |

**数据格式配置**（在 `prepare.py` 中修改）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TEXT_COLUMN` | `question` | CSV 中文本列名 |
| `LABEL_COLUMN` | `是否事实` | CSV 中标签列名 |
| `LABEL_MAP` | `{"事实": "1", "非事实": "0"}` | 标签值 → 模型输出值的映射 |
| `LABEL_DESCRIPTIONS` | `{"1": "事实", "0": "非事实"}` | 模型输出值 → 可读描述 |

Worker 和 Master 均支持 `THINKING` 参数（`enabled` / `disabled` / `auto`），用于控制模型的思考模式。

## 项目结构

```
prompt-optimizer/
├── prepare.py          # 配置文件（模型、数据格式、实验参数、CLI 解析）
├── run.py              # 主程序（优化循环）
├── prompt.md           # Prompt 文件（自动更新）
├── data.csv            # 标注数据（示例）
├── research_log.md     # 实验日志（自动生成）
├── results.tsv         # 结构化实验结果（自动生成）
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
└── LICENSE             # Apache 2.0
```

## 适用场景

本工具适用于任何可以用 LLM + Prompt 解决的**分类任务**，例如：
- 意图识别（事实类 / 非事实类）
- 情感分析（正面 / 负面）
- 内容审核（合规 / 违规）
- 主题分类

只需修改 `prepare.py` 中的 `LABEL_MAP` 和 `data.csv` 即可适配不同任务。

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/prompt-optimizer/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "nlp"，加入技术交流群

<img src="https://github.com/shibing624/TreeSearch/blob/main/docs/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 Prompt Optimizer，请引用：

```bibtex
@software{xu2026promptoptimizer,
  author = {Xu, Ming},
  title = {Prompt Optimizer: Automatic LLM Prompt Optimization with Labeled Data},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shibing624/prompt-optimizer}
}
```

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎贡献！请提交 [Pull Request](https://github.com/shibing624/prompt-optimizer/pulls)。
