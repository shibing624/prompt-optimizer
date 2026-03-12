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
                  ┌──────────────────┐
                  │ 继续迭代 /       │
                  │ Early Stop /     │
                  │ 达到最大轮数      │
                  └──────────────────┘
```

**关键设计：**
- **Train/Val 拆分** — 固定种子拆分，防止 Prompt 过拟合到训练数据
- **增量改写** — Master 只做小幅修改，保留已有效的规则，避免推倒重来
- **Early Stop** — 连续 N 轮 val score 无提升则自动停止，节省 API 调用
- **自动回滚** — val score 下降时回滚到最优 Prompt，在 best 基础上继续优化
- **日志备份** — 每次运行自动备份上一轮的实验日志，方便对比不同轮次

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
WORKER_MODEL_NAME=gpt-4o

# Master 模型（错误分析 + prompt 改写，推荐用更强的模型）
MASTER_API_KEY=your_master_api_key
MASTER_BASE_URL=https://api.openai.com/v1
MASTER_MODEL_NAME=gpt-5
```

支持任何兼容 OpenAI API 格式的模型服务（OpenAI、Azure、本地部署等）。

### 2. 准备数据

准备 CSV 格式的标注数据文件 `data.csv`，包含两列：

| question | 是否事实 |
|---|---|
| 怎么投私信获取线索？ | 事实 |
| 为什么我的流量挣钱越来越少？ | 非事实 |

### 3. 编写初始 Prompt

在 `prompt.md` 中编写初始 Prompt，使用 `{{query}}` 作为用户输入的占位符。

### 4. 运行优化

```bash
python run.py
```

优化过程中会：
- 在终端实时输出每轮的 Train/Val Score
- 自动将最优 Prompt 保存到 `prompt.md`
- 将完整实验历史记录到 `research_log.md`

## 配置说明

在 `prepare.py` 中可调整以下实验参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ITERATIONS` | 2 | 优化迭代轮数 |
| `SAMPLE_SIZE` | 239 | 每轮从 train 集抽样评估的条数 |
| `CONCURRENCY` | 10 | LLM 请求并发数 |
| `VAL_RATIO` | 0.2 | 验证集比例 |
| `PATIENCE` | 3 | 连续 N 轮 val score 无提升则 early stop |

Worker 和 Master 均支持 `THINKING` 参数（`enabled` / `disabled` / `auto`），用于控制模型的思考模式。

## 项目结构

```
prompt-optimizer/
├── prepare.py          # 配置文件（模型参数、实验参数、文件路径）
├── run.py              # 主程序（优化循环）
├── prompt.md           # Prompt 文件（自动更新）
├── data.csv            # 标注数据（示例）
├── research_log.md     # 实验日志（自动生成）
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

只需修改 `data.csv` 的标签列和 `prompt.md` 的初始 Prompt 即可适配不同任务。

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
