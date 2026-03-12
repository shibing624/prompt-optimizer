import os

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

# ====== 实验配置 ======
ITERATIONS = 2         # 优化迭代轮数
SAMPLE_SIZE = 239       # 每轮抽样评估条数（从 train 集抽）
CONCURRENCY = 10        # 分类请求的并发数
VAL_RATIO = 0.2         # 验证集比例（防过拟合）
PATIENCE = 3            # 连续 N 轮 val score 无提升则 early stop

# ====== 文件路径 ======
DATA_FILE = "data.csv"          # GT 标注数据（CSV 格式，列：question, 是否事实）
PROMPT_FILE = "prompt.md"       # AI 自动更新的 prompt 文件
LOG_FILE = "research_log.md"    # 实验日志文件
