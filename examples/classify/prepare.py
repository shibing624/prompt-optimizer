# Classify 示例配置（情感分析二分类）
# 复制到项目根目录覆盖 prepare.py 中的对应配置即可使用

TASK_TYPE = "classify"

TEXT_COLUMNS = ["text"]
LABEL_COLUMN = "label"

LABEL_MAP = {"positive": "positive", "negative": "negative"}
LABEL_DESCRIPTIONS = {
    "positive": "正面情感",
    "negative": "负面情感",
}

PROMPT_VARIABLES = {}  # classify 模式不使用
