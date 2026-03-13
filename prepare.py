# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: prepare for prompt optimization

HUMAN update module
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# ====== Worker 模型配置（执行评测/分类） ======
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_BASE_URL = os.environ.get("WORKER_BASE_URL", "https://api.openai.com/v1")
WORKER_MODEL_NAME = os.environ.get("WORKER_MODEL_NAME", "gpt-4o-mini")
WORKER_THINKING = os.environ.get("WORKER_THINKING", "disabled")    # enabled / disabled / auto

# ====== Master 模型配置（错误分析 + prompt 改写，建议用更大模型） ======
MASTER_API_KEY = os.environ.get("MASTER_API_KEY", "")
MASTER_BASE_URL = os.environ.get("MASTER_BASE_URL", "https://api.openai.com/v1")
MASTER_MODEL_NAME = os.environ.get("MASTER_MODEL_NAME", "gpt-4o")
MASTER_THINKING = os.environ.get("MASTER_THINKING", "disabled")    # enabled / disabled / auto

print(f"WORKER_MODEL_NAME: {WORKER_MODEL_NAME}, api_key={WORKER_API_KEY[:8]}..., base_url={WORKER_BASE_URL}, thinking={WORKER_THINKING}")
print(f"MASTER_MODEL_NAME: {MASTER_MODEL_NAME}, api_key={MASTER_API_KEY[:8]}..., base_url={MASTER_BASE_URL}, thinking={MASTER_THINKING}")

# ====== 任务类型 ======
# "classify" — 单文本分类（Worker 直接输出标签）
# "judge"    — 答案评测（Worker 对比标准答案和模型答案，输出 正确/错误）
TASK_TYPE = "classify"

# ====== 数据格式配置 ======
# classify 模式：TEXT_COLUMNS 只需一个元素，LABEL_COLUMN 是标签列
# judge 模式：TEXT_COLUMNS 包含所有输入字段，LABEL_COLUMN 是 GT 评判结果列
TEXT_COLUMNS = ["text"]        # CSV 中用于构建 prompt 的列名（按顺序）
LABEL_COLUMN = "label"         # CSV 中标签/评判结果列名

# 标签映射：CSV 标签值 → 统一的模型输出值
LABEL_MAP = {"positive": "positive", "negative": "negative"}
# 模型输出值 → 可读描述（用于错误分析展示）
LABEL_DESCRIPTIONS = {
    "positive": "正面情感",
    "negative": "负面情感",
}

# Prompt 模板变量映射（仅 judge 模式使用）
# key = prompt 中的占位符名称（如 {question}），value = CSV 列名
# prompt 中用 {{变量名}} 表示，如 {{question}}、{{standard_answer}}、{{agent_answer}}
PROMPT_VARIABLES = {
    "question": "question",
    "answer": "answer",
}

# ====== 实验配置 ======
ITERATIONS = 5          # 优化迭代轮数（不含初始评测）
TRAIN_SAMPLE_SIZE = 200 # 每轮 train 抽样条数
VAL_SAMPLE_SIZE = 100   # 每轮 valid 抽样条数
CONCURRENCY = 10        # 分类/评测请求的并发数
PATIENCE = 2            # 连续 N 轮 valid score 无提升则 early stop
VOTE_COUNT = 1          # 多数投票次数（1 = 不投票，3/5 = 多数投票减少随机性）
PRIMARY_METRIC = "f1"   # 主评估指标（accuracy / f1 / precision / recall）
MAX_RETRIES = 3         # LLM 调用失败时的最大重试次数

# ====== 文件路径 ======
DATA_FILE = "data.csv"         # GT 标注数据
PROMPT_FILE = "prompt.md"     # AI 自动更新的 prompt 文件
LOG_FILE = "research_log.md"  # 实验日志文件
RESULTS_FILE = "results.tsv"  # 结构化实验结果记录


def parse_args():
    """解析命令行参数，覆盖 prepare.py 中的默认值"""
    parser = argparse.ArgumentParser(description="Prompt Optimizer")
    parser.add_argument("--iterations", type=int, default=None, help="优化迭代轮数")
    parser.add_argument("--train-sample-size", type=int, default=None, help="每轮 train 抽样条数")
    parser.add_argument("--val-sample-size", type=int, default=None, help="每轮 valid 抽样条数")
    parser.add_argument("--concurrency", type=int, default=None, help="LLM 请求并发数")
    parser.add_argument("--patience", type=int, default=None, help="Early stop 耐心值")
    parser.add_argument("--vote-count", type=int, default=None, help="多数投票次数（1=不投票）")
    parser.add_argument("--metric", type=str, default=None,
                        choices=["accuracy", "f1", "precision", "recall"], help="主评估指标")
    parser.add_argument("--data", type=str, default=None, help="标注数据文件路径")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt 文件路径")
    parser.add_argument("--task-type", type=str, default=None,
                        choices=["classify", "judge"], help="任务类型")
    return parser.parse_args()


def apply_args(args):
    """将命令行参数应用到全局配置"""
    global ITERATIONS, TRAIN_SAMPLE_SIZE, VAL_SAMPLE_SIZE, CONCURRENCY, PATIENCE
    global VOTE_COUNT, PRIMARY_METRIC, DATA_FILE, PROMPT_FILE, TASK_TYPE
    if args.iterations is not None:
        ITERATIONS = args.iterations
    if args.train_sample_size is not None:
        TRAIN_SAMPLE_SIZE = args.train_sample_size
    if args.val_sample_size is not None:
        VAL_SAMPLE_SIZE = args.val_sample_size
    if args.concurrency is not None:
        CONCURRENCY = args.concurrency
    if args.patience is not None:
        PATIENCE = args.patience
    if args.vote_count is not None:
        VOTE_COUNT = args.vote_count
    if args.metric is not None:
        PRIMARY_METRIC = args.metric
    if args.data is not None:
        DATA_FILE = args.data
    if args.prompt is not None:
        PROMPT_FILE = args.prompt
    if args.task_type is not None:
        TASK_TYPE = args.task_type
