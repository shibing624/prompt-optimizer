import os
import csv
import random
import time
import subprocess
import openai
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from loguru import logger
from tqdm import tqdm
from prepare import (
    WORKER_API_KEY, WORKER_BASE_URL, WORKER_MODEL_NAME, WORKER_THINKING,
    MASTER_API_KEY, MASTER_BASE_URL, MASTER_MODEL_NAME, MASTER_THINKING,
    ITERATIONS, TRAIN_SAMPLE_SIZE, VAL_SAMPLE_SIZE, CONCURRENCY,
    PATIENCE, VOTE_COUNT, PRIMARY_METRIC, MAX_RETRIES,
    TASK_TYPE, TEXT_COLUMNS, LABEL_COLUMN, LABEL_MAP, LABEL_DESCRIPTIONS,
    PROMPT_VARIABLES,
    DATA_FILE, PROMPT_FILE, LOG_FILE, RESULTS_FILE,
    parse_args, apply_args,
)

worker_client = openai.OpenAI(api_key=WORKER_API_KEY, base_url=WORKER_BASE_URL)
master_client = openai.OpenAI(api_key=MASTER_API_KEY, base_url=MASTER_BASE_URL)


# ====== 数据与文件加载 ======
def load_data(path):
    """加载 CSV 标注数据，支持 classify（单文本列）和 judge（多文本列）两种模式"""
    data = []
    if not os.path.exists(path):
        logger.warning(f"{path} not found. Returning empty dataset.")
        return data

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 提取所有文本列
            fields = {}
            skip = False
            for col in TEXT_COLUMNS:
                val = row.get(col, "").strip()
                if not val and col == TEXT_COLUMNS[0]:
                    skip = True
                    break
                fields[col] = val
            if skip:
                continue

            # 提取标签
            label_str = row.get(LABEL_COLUMN, "").strip()
            label = LABEL_MAP.get(label_str, label_str)

            # 兼容 classify 模式：保留 "text" 字段
            if TASK_TYPE == "classify":
                fields["text"] = fields.get(TEXT_COLUMNS[0], "")

            data.append({"fields": fields, "label": label})
    return data


def sample_train_val(data, train_size, val_size):
    """从全量数据中抽取不重叠的 train 和 valid 样本"""
    total_needed = train_size + val_size
    if total_needed > len(data):
        logger.warning(f"所需样本数({total_needed})超过数据总量({len(data)})，按比例缩减")
        ratio = len(data) / total_needed
        train_size = max(1, int(train_size * ratio))
        val_size = max(1, len(data) - train_size)
        total_needed = train_size + val_size

    sampled = random.sample(data, total_needed)
    train_set = sampled[:train_size]
    val_set = sampled[train_size:]
    logger.info(f"抽样完成: train={len(train_set)}, valid={len(val_set)}, 总数据={len(data)}")
    return train_set, val_set


def _get_default_prompt():
    """根据任务类型生成默认 prompt"""
    if TASK_TYPE == "judge":
        label_options = "/".join(sorted(LABEL_MAP.values()))
        return f"""你是一个专业的RAG问答质量评测员。你需要评判：基于给定的知识片段（RAG检索结果），模型的回答质量如何。

知识片段（RAG检索结果）：
{{{{knowledge}}}}

用户问题：
{{{{question}}}}

模型答案：
{{{{agent_answer}}}}

直接输出一个标签：{label_options}

不要输出任何解释、理由或其他内容。"""
    else:
        label_desc = "、".join([f"{v}({k})" for k, v in LABEL_DESCRIPTIONS.items()])
        return f"你是分类专家。判断用户输入属于哪个类别，仅输出类别标签（{label_desc}）。\n\n用户输入如下：\n{{{{query}}}}"


def read_prompt(path):
    if not os.path.exists(path):
        base_prompt = _get_default_prompt()
        write_prompt(path, base_prompt)
        return base_prompt
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_prompt(path, prompt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)


def append_to_log(path, step, prompt, metrics, analysis):
    with open(path, "a", encoding="utf-8") as f:
        metrics_str = " | ".join([f"**{k}:** {v:.4f}" for k, v in metrics.items()])
        f.write(f"## Step {step}\n")
        f.write(f"{metrics_str}\n\n")
        f.write(f"**Prompt ({len(prompt.splitlines())} lines):**\n```markdown\n{prompt}\n```\n\n")
        f.write(f"**Error Analysis:**\n{analysis}\n\n")
        f.write("---\n\n")


def append_to_results(path, step, metrics_train, metrics_val, prompt_lines, status, description=""):
    """追加结构化实验结果到 results.tsv"""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["step", "train_accuracy", "train_f1", "val_accuracy", "val_f1",
                             "prompt_lines", "status", "timestamp", "description"])
        writer.writerow([
            step,
            f"{metrics_train.get('accuracy', 0):.4f}",
            f"{metrics_train.get('f1', 0):.4f}",
            f"{metrics_val.get('accuracy', 0):.4f}",
            f"{metrics_val.get('f1', 0):.4f}",
            prompt_lines,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description,
        ])


# ====== Git 集成 ======
def git_commit(step, val_score, status):
    """每轮自动 git commit prompt 文件"""
    try:
        subprocess.run(["git", "add", PROMPT_FILE, RESULTS_FILE], capture_output=True, check=True)
        msg = f"step {step}: val_{PRIMARY_METRIC}={val_score:.4f} ({status})"
        subprocess.run(["git", "commit", "-m", msg], capture_output=True, check=True)
        logger.info(f"Git commit: {msg}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


# ====== 调用 LLM（带重试） ======
def _build_extra_body(thinking_type):
    if thinking_type in ("enabled", "auto"):
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    return {}


def llm_worker(prompt, temperature=0.0):
    for attempt in range(MAX_RETRIES):
        try:
            resp = worker_client.chat.completions.create(
                model=WORKER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **_build_extra_body(WORKER_THINKING),
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Worker LLM retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Worker LLM API Error (all retries failed): {e}")
                return ""


def llm_master(prompt, temperature=0.7):
    for attempt in range(MAX_RETRIES):
        try:
            resp = master_client.chat.completions.create(
                model=MASTER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **_build_extra_body(MASTER_THINKING),
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Master LLM retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Master LLM API Error (all retries failed): {e}")
                return ""


# ====== 解析预测结果 ======
def _parse_prediction(out):
    """从 LLM 输出中解析预测标签"""
    if not out or not out.strip():
        logger.warning("LLM 返回空输出，标记为 PARSE_FAIL")
        return "PARSE_FAIL"

    out_clean = out.strip()
    all_labels = sorted(LABEL_MAP.values())

    # 1. 完整匹配（输出就是标签本身）
    if out_clean in all_labels:
        return out_clean

    # 2. 逐行扫描，取第一个包含标签的行（优先靠前的内容）
    for line in out_clean.split("\n"):
        line = line.strip()
        found = [lb for lb in all_labels if lb in line]
        if len(found) == 1:
            return found[0]

    # 3. 全文包含唯一标签
    found = [lb for lb in all_labels if lb in out_clean]
    if len(found) == 1:
        return found[0]

    # 解析失败
    logger.warning(f"无法解析 LLM 输出: {out_clean[:80]}")
    return "PARSE_FAIL"


# ====== 构建单条 prompt ======
def _build_query(prompt, item):
    """根据 TASK_TYPE 构建完整的 LLM 输入"""
    fields = item["fields"]

    if TASK_TYPE == "judge":
        # Judge 模式：用 PROMPT_VARIABLES 映射替换 prompt 中的 {{变量名}}
        query = prompt
        for var_name, col_name in PROMPT_VARIABLES.items():
            placeholder = "{{" + var_name + "}}"
            query = query.replace(placeholder, fields.get(col_name, ""))
        return query
    else:
        # Classify 模式：用 {{query}} 占位符替换
        text = fields.get("text", fields.get(TEXT_COLUMNS[0], ""))
        if "{{query}}" in prompt:
            query = prompt.replace("{{query}}", text)
        else:
            query = f"{prompt}\n\n用户输入如下：\n{text}"
        # 追加输出格式约束
        all_labels = set(LABEL_MAP.values())
        label_str = "/".join(sorted(all_labels))
        query += f"\n\n请直接输出{label_str}，不需要任何解释。"
        return query


# ====== 运行评测（并发 + 多数投票） ======
def _evaluate_single(idx, item, prompt):
    query = _build_query(prompt, item)

    if VOTE_COUNT <= 1:
        out = llm_worker(query, temperature=0.0)
        pred = _parse_prediction(out)
    else:
        votes = []
        for _ in range(VOTE_COUNT):
            out = llm_worker(query, temperature=0.3)
            votes.append(_parse_prediction(out))
        pred = Counter(votes).most_common(1)[0][0]

    # 构建摘要文本用于日志显示
    display_text = item["fields"].get(TEXT_COLUMNS[0], "")[:30]
    return idx, display_text, pred, item["label"]


def run_prompt(prompt, dataset, desc="Eval"):
    preds = [None] * len(dataset)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_idx = {
            executor.submit(_evaluate_single, idx, item, prompt): idx
            for idx, item in enumerate(dataset)
        }

        completed_count = 0
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc=desc):
            idx, display_text, pred, gt = future.result()
            preds[idx] = pred
            completed_count += 1

            if completed_count % 5 == 0:
                logger.info(f"[{desc}] {completed_count}/{len(dataset)} | {display_text}... | Pred: {pred} | GT: {gt}")

    return preds


# ====== 多指标评估 ======
def evaluate(preds, dataset):
    """返回 (metrics_dict, errors_list)"""
    if not dataset:
        return {}, []
    gt = [x["label"] for x in dataset]

    all_labels = sorted(set(LABEL_MAP.values()))
    metrics = {
        "accuracy": accuracy_score(gt, preds),
        "f1": f1_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
        "precision": precision_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
        "recall": recall_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
    }

    errors = []
    for i, (p, g) in enumerate(zip(preds, gt)):
        if p != g:
            errors.append({
                "fields": dataset[i]["fields"],
                "display_text": dataset[i]["fields"].get(TEXT_COLUMNS[0], "")[:100],
                "expected": g,
                "predicted": p,
            })
    return metrics, errors


# ====== 错误样本格式化 ======
MAX_ERROR_SAMPLES = 10
MAX_FIELD_LEN = 300


def _format_errors(val_errors, train_errors):
    """合并 train/val 错误，统一随机采样并格式化"""
    all_errors = val_errors + train_errors
    if not all_errors:
        return "无错误样本。", 0

    sampled = random.sample(all_errors, min(len(all_errors), MAX_ERROR_SAMPLES))
    lines = []
    for i, e in enumerate(sampled, 1):
        if TASK_TYPE == "judge":
            parts = [f"[错误样本 #{i}] Expected={e['expected']} Predicted={e['predicted']}"]
            for var_name, col_name in PROMPT_VARIABLES.items():
                val = e["fields"].get(col_name, "")
                if len(val) > MAX_FIELD_LEN:
                    val = val[:MAX_FIELD_LEN] + "..."
                parts.append(f"  {var_name}: {val}")
            lines.append("\n".join(parts))
        else:
            lines.append(f"[错误样本 #{i}] Text: {e['display_text']} | Expected={e['expected']} Predicted={e['predicted']}")

    return "\n\n".join(lines), len(all_errors)


# ====== prompt 改写（直接基于原始错误样本） ======
def improve_prompt(prompt, metrics_train, metrics_val, val_errors, train_errors, history):
    """分析错误 + 改进 prompt 一步完成，不经过中间层转述"""
    error_str, total_errors = _format_errors(val_errors, train_errors)

    history_str = ""
    for h in history[-3:]:
        history_str += (f"- Step {h['step']}: Train={h['train_metrics'].get(PRIMARY_METRIC, 0):.4f}, "
                        f"Valid={h['val_metrics'].get(PRIMARY_METRIC, 0):.4f}, "
                        f"errors={h['total_errors']}\n")

    label_desc_str = ", ".join([f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()])
    all_labels = "/".join(sorted(LABEL_MAP.values()))

    if TASK_TYPE == "judge":
        task_desc = f"RAG问答质量评测（Judge）任务：基于知识片段判断模型答案质量，输出 {all_labels}。"
        var_list = ", ".join([f"{{{{{k}}}}}" for k in PROMPT_VARIABLES.keys()])
        placeholder_note = f"prompt 中必须保留以下变量占位符：{var_list}"
    else:
        task_desc = f"分类任务：标签含义 — {label_desc_str}。模型仅输出 {all_labels}。"
        placeholder_note = "prompt 末尾必须保留 {{query}} 占位符"

    query = f"""你是一个专业的 prompt 工程师。你的目标是在当前 prompt 基础上做**增量改进**，提升评测准确率。

任务背景：{task_desc}

近几轮实验历史：
{history_str}

当前 Prompt（需改进）：
{prompt}

当前分数：Train {PRIMARY_METRIC}={metrics_train.get(PRIMARY_METRIC, 0):.4f}, Valid {PRIMARY_METRIC}={metrics_val.get(PRIMARY_METRIC, 0):.4f}
总错误数：{total_errors}（以下展示随机采样的 {min(total_errors, MAX_ERROR_SAMPLES)} 条）

错误样本（Expected=标注答案，Predicted=模型输出）：
{error_str}

===== 改写要求 =====

1. **先分析错误模式**：找出错误样本中的共性规律，而非逐个修复
2. **增量修改，不要重写**：在当前 prompt 基础上做精准补充/修正，保留有效规则和结构
3. **严禁过拟合**：不要把具体案例原文当规则，提炼通用判定原则
4. **{placeholder_note}**

只返回完整的修改后 prompt 文本，不要包含 markdown 代码块标记或任何解释。"""

    new_prompt = llm_master(query).strip()
    if new_prompt.startswith("```") and new_prompt.endswith("```"):
        new_prompt = "\n".join(new_prompt.split("\n")[1:-1]).strip()
    if new_prompt.startswith("```markdown"):
        new_prompt = "\n".join(new_prompt.split("\n")[1:]).strip()
        if new_prompt.endswith("```"):
            new_prompt = new_prompt[:-3].strip()

    # 检查占位符完整性
    if TASK_TYPE == "judge":
        for var_name in PROMPT_VARIABLES.keys():
            placeholder = "{{" + var_name + "}}"
            if placeholder not in new_prompt:
                logger.warning(f"Missing placeholder {placeholder} in new prompt, appending.")
                new_prompt += f"\n\n{var_name}: {placeholder}"
    else:
        if "{{query}}" not in new_prompt:
            new_prompt += "\n\n---\n用户输入如下：\n{{query}}"

    return new_prompt


# ====== 优化循环 ======
def optimize():
    args = parse_args()
    apply_args(args)

    logger.info(f"任务类型: {TASK_TYPE}")
    data = load_data(DATA_FILE)
    if not data:
        logger.error("No data available to run optimization.")
        return

    # 备份旧日志
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        backup_name = LOG_FILE.replace(".md", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        os.rename(LOG_FILE, backup_name)
        logger.info(f"旧日志已备份为: {backup_name}")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Prompt Optimization Log\n\n")

    current_prompt = read_prompt(PROMPT_FILE)
    best_prompt = current_prompt
    best_val_score = 0.0
    no_improve_count = 0
    history = []

    logger.info(f"配置: task_type={TASK_TYPE}, iterations={ITERATIONS}, patience={PATIENCE}, "
                f"metric={PRIMARY_METRIC}, vote_count={VOTE_COUNT}, "
                f"train_sample={TRAIN_SAMPLE_SIZE}, val_sample={VAL_SAMPLE_SIZE}")

    # ====== Step 0: 评测初始 prompt（baseline） ======
    def _eval_prompt(prompt, step_label):
        """评测指定 prompt，从全量数据抽取不重叠的 train/valid 样本"""
        train_sample, val_sample = sample_train_val(data, TRAIN_SAMPLE_SIZE, VAL_SAMPLE_SIZE)
        train_preds = run_prompt(prompt, train_sample, desc=f"{step_label} Train")
        m_train, t_errors = evaluate(train_preds, train_sample)

        val_preds = run_prompt(prompt, val_sample, desc=f"{step_label} Valid")
        m_val, v_errors = evaluate(val_preds, val_sample)
        return m_train, m_val, t_errors, v_errors

    logger.info("=== Step 0: 评测初始 Prompt (Baseline) ===")
    metrics_train, metrics_val, train_errors, val_errors = _eval_prompt(current_prompt, "Step0")

    val_primary = metrics_val.get(PRIMARY_METRIC, 0)
    train_primary = metrics_train.get(PRIMARY_METRIC, 0)
    metrics_str = " | ".join([f"{k}: Train={metrics_train[k]:.4f}/Valid={metrics_val[k]:.4f}" for k in metrics_val])
    logger.info(f"Baseline: {metrics_str} | Prompt Lines: {len(current_prompt.splitlines())}")

    best_val_score = val_primary
    best_prompt = current_prompt
    append_to_results(RESULTS_FILE, 0, metrics_train, metrics_val,
                      len(current_prompt.splitlines()), "baseline")
    git_commit(0, val_primary, "baseline")

    if val_primary == 1.0 and train_primary == 1.0:
        logger.info("初始 prompt 已达满分，无需优化。")
        append_to_log(LOG_FILE, 0, current_prompt, metrics_val, "Perfect score on baseline.")
    else:
        # ====== 迭代优化循环（每轮：基于错误样本生成新 prompt → 评测新 prompt） ======
        for step in range(1, ITERATIONS + 1):
            logger.info(f"--- Iteration {step}/{ITERATIONS}: 生成新 prompt ---")

            total_errors = len(train_errors) + len(val_errors)
            logger.info(f"错误样本: train={len(train_errors)}, valid={len(val_errors)}, total={total_errors}")

            append_to_log(LOG_FILE, step - 1, current_prompt, metrics_val,
                          f"errors: train={len(train_errors)}, valid={len(val_errors)}")
            history.append({
                "step": step - 1,
                "prompt": current_prompt,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
                "total_errors": total_errors,
            })

            # 直接基于原始错误样本生成新 prompt
            logger.info("Generating new prompt (Master)...")
            new_prompt = improve_prompt(current_prompt, metrics_train, metrics_val,
                                        val_errors, train_errors, history)
            write_prompt(PROMPT_FILE, new_prompt)
            current_prompt = new_prompt
            logger.info(f"New prompt written ({len(new_prompt.splitlines())} lines)")

            # 评测新 prompt
            logger.info(f"--- Iteration {step}/{ITERATIONS}: 评测新 prompt ---")
            metrics_train, metrics_val, train_errors, val_errors = _eval_prompt(current_prompt, f"Step{step}")

            val_primary = metrics_val.get(PRIMARY_METRIC, 0)
            train_primary = metrics_train.get(PRIMARY_METRIC, 0)
            metrics_str = " | ".join([f"{k}: Train={metrics_train[k]:.4f}/Valid={metrics_val[k]:.4f}" for k in metrics_val])
            logger.info(f"{metrics_str} | Prompt Lines: {len(current_prompt.splitlines())}")

            if val_primary > best_val_score:
                best_val_score = val_primary
                best_prompt = current_prompt
                no_improve_count = 0
                status = "keep"
                logger.info(f"Valid {PRIMARY_METRIC} improved! Best: {best_val_score:.4f}")
            else:
                no_improve_count += 1
                status = "discard"
                logger.warning(f"Valid {PRIMARY_METRIC} not improved ({no_improve_count}/{PATIENCE}). Best: {best_val_score:.4f}")
                current_prompt = best_prompt
                logger.info("Rolled back to best prompt.")

            append_to_results(RESULTS_FILE, step, metrics_train, metrics_val,
                              len(current_prompt.splitlines()), status)
            git_commit(step, val_primary, status)

            if val_primary == 1.0 and train_primary == 1.0:
                logger.info("Perfect score. Stopping.")
                append_to_log(LOG_FILE, step, current_prompt, metrics_val, "Perfect score reached.")
                break

            if no_improve_count >= PATIENCE:
                logger.warning(f"Early stopping: valid {PRIMARY_METRIC} 连续 {PATIENCE} 轮未提升。")
                write_prompt(PROMPT_FILE, best_prompt)
                append_to_log(LOG_FILE, step, current_prompt, metrics_val,
                              f"Early stop. Best valid {PRIMARY_METRIC}={best_val_score:.4f}.")
                append_to_results(RESULTS_FILE, step, metrics_train, metrics_val,
                                  len(best_prompt.splitlines()), "early_stop", "rolled back to best")
                git_commit(step, best_val_score, "early_stop")
                break

    logger.info("=== OPTIMIZATION COMPLETE ===")
    logger.info(f"BEST VALID {PRIMARY_METRIC.upper()}: {best_val_score:.4f}")
    write_prompt(PROMPT_FILE, best_prompt)
    logger.info(f"Best prompt saved to {PROMPT_FILE}")
    logger.info(f"Experiment log: {LOG_FILE} | Results: {RESULTS_FILE}")


if __name__ == "__main__":
    optimize()
