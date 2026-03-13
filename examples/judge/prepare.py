# Judge 示例配置（RAG 问答质量评测三分类）
# 复制到项目根目录覆盖 prepare.py 中的对应配置即可使用

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
