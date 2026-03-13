import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# ====== Worker 模型配置（分类执行） ======
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_BASE_URL = os.environ.get("WORKER_BASE_URL", "https://api.openai.com/v1")
WORKER_MODEL_NAME = os.environ.get("WORKER_MODEL_NAME", "gpt-4o-mini")
WORKER_THINKING = os.environ.get("WORKER_THINKING", "disabled")    # enabled / disabled / auto

# ====== Master 模型配置（错误分析 + prompt 改写，建议用更大模型） ======
MASTER_API_KEY = os.environ.get("MASTER_API_KEY", "")
MASTER_BASE_URL = os.environ.get("MASTER_BASE_URL", "https://api.openai.com/v1")
MASTER_MODEL_NAME = os.environ.get("MASTER_MODEL_NAME", "gpt-4o")
MASTER_THINKING = os.environ.get("MASTER_THINKING", "disabled")    # enabled / disabled / auto

# ====== 数据格式配置（适配不同分类任务） ======
TEXT_COLUMN = "question"                # CSV 中文本列名
LABEL_COLUMN = "是否事实"               # CSV 中标签列名
LABEL_MAP = {"事实": "1", "非事实": "0"}  # 标签值 → 模型输出值的映射
LABEL_DESCRIPTIONS = {"1": "事实", "0": "非事实"}  # 模型输出值 → 可读描述（用于错误分析）

# ====== 实验配置 ======
ITERATIONS = 5         # 优化迭代轮数
SAMPLE_SIZE = 200       # 每轮抽样评估条数（从 train 集抽）
CONCURRENCY = 10        # 分类请求的并发数
VAL_RATIO = 0.2         # 验证集比例（防过拟合）
PATIENCE = 3            # 连续 N 轮 val score 无提升则 early stop
VOTE_COUNT = 1          # 多数投票次数（1 = 不投票，3/5 = 多数投票减少随机性）
PRIMARY_METRIC = "accuracy"  # 主评估指标，作为 keep/discard 决策依据（accuracy / f1 / precision / recall）
MAX_RETRIES = 3         # LLM 调用失败时的最大重试次数

# ====== 文件路径 ======
DATA_FILE = "data.csv"          # GT 标注数据
PROMPT_FILE = "prompt.md"       # AI 自动更新的 prompt 文件
LOG_FILE = "research_log.md"    # 实验日志文件
RESULTS_FILE = "results.tsv"    # 结构化实验结果记录


def parse_args():
    """解析命令行参数，覆盖 prepare.py 中的默认值"""
    parser = argparse.ArgumentParser(description="Prompt Optimizer")
    parser.add_argument("--iterations", type=int, default=None, help="优化迭代轮数")
    parser.add_argument("--sample-size", type=int, default=None, help="每轮抽样评估条数")
    parser.add_argument("--concurrency", type=int, default=None, help="LLM 请求并发数")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--patience", type=int, default=None, help="Early stop 耐心值")
    parser.add_argument("--vote-count", type=int, default=None, help="多数投票次数（1=不投票）")
    parser.add_argument("--metric", type=str, default=None, choices=["accuracy", "f1", "precision", "recall"], help="主评估指标")
    parser.add_argument("--data", type=str, default=None, help="标注数据文件路径")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt 文件路径")
    return parser.parse_args()


def apply_args(args):
    """将命令行参数应用到全局配置"""
    global ITERATIONS, SAMPLE_SIZE, CONCURRENCY, VAL_RATIO, PATIENCE
    global VOTE_COUNT, PRIMARY_METRIC, DATA_FILE, PROMPT_FILE
    if args.iterations is not None:
        ITERATIONS = args.iterations
    if args.sample_size is not None:
        SAMPLE_SIZE = args.sample_size
    if args.concurrency is not None:
        CONCURRENCY = args.concurrency
    if args.val_ratio is not None:
        VAL_RATIO = args.val_ratio
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
