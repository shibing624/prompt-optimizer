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
                    │  Train/Val 抽样  │  每轮随机抽样，不重叠
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
                  │  Master 分析错误    │  直接基于原始错误样本
                  │  + 增量改写 Prompt  │  一步完成，无中间层
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
- **Train/Val 随机抽样** — 每轮从全量数据随机抽取不重叠的 train/val 样本
- **原始错误驱动** — 直接把原始错误样本喂给 Master，一步完成分析+改写，无二次转述损失
- **增量改写** — Master 只做小幅修改，保留已有效的规则，避免推倒重来
- **多指标评估** — 同时计算 Accuracy、F1（macro）、Precision、Recall，支持选择主指标
- **多数投票** — 同一 query 可跑多次取多数票，减少 LLM 输出随机性
- **自动回滚** — Val score 下降时回滚到最优 Prompt，在 best 基础上继续优化
- **Git 集成** — 每轮自动 commit prompt 文件，方便回溯任何历史版本
- **结构化记录** — `results.tsv` 记录每轮指标，可程序化分析
- **Early Stop** — 连续 N 轮无提升则自动停止，节省 API 调用
- **异常重试** — LLM 调用失败自动指数退避重试

## 支持两种任务模式

| 模式 | 说明 | 示例 |
|------|------|------|
| **classify** | 单文本分类，Worker 直接输出标签 | 情感分析、意图识别、内容审核 |
| **judge** | 多字段评测，Worker 对比多个输入字段后输出判定 | RAG 问答质量评测、翻译质量评测 |

## 安装

```bash
git clone https://github.com/shibing624/prompt-optimizer.git
cd prompt-optimizer
pip install -r requirements.txt
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key 和模型配置：

```env
# Worker 模型（执行分类/评测任务，推荐用快速且便宜的模型）
WORKER_API_KEY=your_worker_api_key
WORKER_BASE_URL=https://api.openai.com/v1
WORKER_MODEL_NAME=gpt-4o-mini

# Master 模型（分析错误 + 改写 prompt，推荐用更强的模型）
MASTER_API_KEY=your_master_api_key
MASTER_BASE_URL=https://api.openai.com/v1
MASTER_MODEL_NAME=gpt-4o
```

支持任何兼容 OpenAI API 格式的模型服务（OpenAI、Azure、DeepSeek、本地部署等）。

### 2. 用示例数据跑一下

项目自带两个示例，可以直接运行：

**示例 1：情感分类（classify 模式）**

```bash
# 复制示例数据和 prompt 到根目录
cp examples/classify/data.csv data.csv
cp examples/classify/prompt.md prompt.md

# 运行（默认配置即为 classify 模式）
python run.py --iterations 3 --train-sample-size 20 --val-sample-size 10
```

**示例 2：RAG 问答评测（judge 模式）**

```bash
# 复制示例数据和 prompt
cp examples/judge/data.csv data.csv
cp examples/judge/prompt.md prompt.md
```

然后修改 `prepare.py` 中的数据格式配置（可直接参考 `examples/judge/prepare.py`）：

```python
TASK_TYPE = "judge"
TEXT_COLUMNS = ["knowledge", "question", "answer"]
LABEL_COLUMN = "label"
LABEL_MAP = {"正确": "正确", "错误": "错误", "无效": "无效"}
LABEL_DESCRIPTIONS = {
    "正确": "模型答案基于知识片段正确回答了问题",
    "错误": "知识片段与问题相关，但模型答案存在关键错误",
    "无效": "知识片段与问题不相关，无法据此判断答案质量",
}
PROMPT_VARIABLES = {
    "knowledge": "knowledge",
    "question": "question",
    "answer": "answer",
}
```

```bash
python run.py --iterations 3 --train-sample-size 10 --val-sample-size 5
```

### 3. 用你自己的数据

1. 准备 CSV 格式的标注数据文件 `data.csv`
2. 修改 `prepare.py` 中的数据格式配置（`TEXT_COLUMNS`、`LABEL_COLUMN`、`LABEL_MAP` 等）
3. 编写初始 Prompt 到 `prompt.md`（classify 模式用 `{{query}}` 占位符，judge 模式用自定义变量占位符）
4. 运行 `python run.py`

### 4. 命令行参数

```bash
python run.py --iterations 10 --patience 3 --metric f1 --vote-count 3 --concurrency 20
```

所有参数均可通过命令行覆盖 `prepare.py` 中的默认值。

## 配置说明

在 `prepare.py` 中可调整以下参数：

| 参数 | 默认值 | CLI 参数 | 说明 |
|------|--------|----------|------|
| `TASK_TYPE` | classify | `--task-type` | 任务模式（classify / judge） |
| `ITERATIONS` | 5 | `--iterations` | 优化迭代轮数 |
| `TRAIN_SAMPLE_SIZE` | 200 | `--train-sample-size` | 每轮 train 抽样条数 |
| `VAL_SAMPLE_SIZE` | 100 | `--val-sample-size` | 每轮 valid 抽样条数 |
| `CONCURRENCY` | 10 | `--concurrency` | LLM 请求并发数 |
| `PATIENCE` | 2 | `--patience` | 连续 N 轮无提升则 early stop |
| `VOTE_COUNT` | 1 | `--vote-count` | 多数投票次数（1=不投票，3/5=投票） |
| `PRIMARY_METRIC` | f1 | `--metric` | 主评估指标（accuracy/f1/precision/recall） |
| `MAX_RETRIES` | 3 | — | LLM 调用失败时的最大重试次数 |

**数据格式配置**（在 `prepare.py` 中修改）：

| 参数 | 说明 |
|------|------|
| `TEXT_COLUMNS` | CSV 中用于构建 prompt 的列名列表 |
| `LABEL_COLUMN` | CSV 中标签列名 |
| `LABEL_MAP` | CSV 标签值 → 模型输出值的映射 |
| `LABEL_DESCRIPTIONS` | 模型输出值 → 可读描述 |
| `PROMPT_VARIABLES` | Prompt 模板变量 → CSV 列名映射（仅 judge 模式） |

Worker 和 Master 均支持 `THINKING` 参数（`enabled` / `disabled` / `auto`），用于控制模型的思考模式。

## 项目结构

```
prompt-optimizer/
├── prepare.py              # 配置文件（模型、数据格式、实验参数、CLI 解析）
├── run.py                  # 主程序（优化循环）
├── prompt.md               # Prompt 文件（自动更新）
├── research_log.md         # 实验日志（自动生成）
├── examples/
│   ├── classify/           # 情感分类示例
│   │   ├── data.csv        #   30 条标注数据
│   │   ├── prompt.md       #   初始 prompt
│   │   └── prepare.py      #   配置参考
│   └── judge/              # RAG 问答评测示例
│       ├── data.csv        #   15 条标注数据
│       ├── prompt.md       #   初始 prompt
│       └── prepare.py      #   配置参考
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量示例
└── LICENSE                 # Apache 2.0
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `prompt.md` | 最优 Prompt（每轮自动更新） |
| `results.tsv` | 结构化实验结果（每轮 train/val 指标） |
| `research_log.md` | 完整实验日志（prompt + 指标 + 错误分析） |
| Git 历史 | 每轮自动 commit，可 `git log` 查看 prompt 演化 |

## 适用场景

本工具适用于任何可以用 LLM + Prompt 解决的分类/评测任务，例如：
- 情感分析（positive / negative）
- 意图识别（事实类 / 非事实类）
- 内容审核（合规 / 违规）
- 主题分类
- RAG 问答质量评测（正确 / 错误 / 无效）
- 翻译质量评测

只需修改 `prepare.py` 中的配置和 `data.csv` 即可适配不同任务。

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
