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
    ITERATIONS, SAMPLE_SIZE, CONCURRENCY,
    VAL_RATIO, PATIENCE, VOTE_COUNT, PRIMARY_METRIC, MAX_RETRIES,
    TEXT_COLUMN, LABEL_COLUMN, LABEL_MAP, LABEL_DESCRIPTIONS,
    DATA_FILE, PROMPT_FILE, LOG_FILE, RESULTS_FILE,
    parse_args, apply_args,
)

worker_client = openai.OpenAI(api_key=WORKER_API_KEY, base_url=WORKER_BASE_URL)
master_client = openai.OpenAI(api_key=MASTER_API_KEY, base_url=MASTER_BASE_URL)


# ====== 数据与文件加载 ======
def load_data(path):
    """加载 CSV 标注数据，自动适配 TEXT_COLUMN / LABEL_COLUMN / LABEL_MAP"""
    data = []
    if not os.path.exists(path):
        logger.warning(f"{path} not found. Returning empty dataset.")
        return data

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(TEXT_COLUMN, "").strip()
            label_str = row.get(LABEL_COLUMN, "").strip()
            if not text:
                continue
            label = LABEL_MAP.get(label_str, "0")
            data.append({"text": text, "label": label})
    return data


def split_train_val(data, val_ratio):
    """固定随机种子拆分 train/val，保证每次运行拆分一致"""
    shuffled = data[:]
    random.Random(42).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    logger.info(f"数据拆分完成: train={len(train_set)}, val={len(val_set)}")
    return train_set, val_set


def read_prompt(path):
    if not os.path.exists(path):
        label_desc = "、".join([f"{v}({k})" for k, v in LABEL_DESCRIPTIONS.items()])
        base_prompt = f"你是分类专家。判断用户输入属于哪个类别，仅输出类别标签（{label_desc}）。\n\n用户输入如下：\n{{{{query}}}}"
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


# ====== 运行分类（并发 + 多数投票） ======
def _parse_prediction(out):
    """从 LLM 输出中解析预测标签"""
    out_clean = out.strip()
    all_labels = set(LABEL_MAP.values())
    found = [lb for lb in all_labels if lb in out_clean]
    if len(found) == 1:
        return found[0]
    if out_clean and out_clean[-1] in all_labels:
        return out_clean[-1]
    return "0"


def _classify_single(idx, item, prompt):
    text = item["text"]
    if "{{query}}" in prompt:
        query = prompt.replace("{{query}}", text)
    else:
        query = f"{prompt}\n\n用户输入如下：\n{text}"

    all_labels = set(LABEL_MAP.values())
    label_str = "/".join(sorted(all_labels))
    query += f"\n\n请直接输出{label_str}，不需要任何解释。"

    if VOTE_COUNT <= 1:
        out = llm_worker(query, temperature=0.0)
        pred = _parse_prediction(out)
    else:
        votes = []
        for _ in range(VOTE_COUNT):
            out = llm_worker(query, temperature=0.3)
            votes.append(_parse_prediction(out))
        pred = Counter(votes).most_common(1)[0][0]

    return idx, text, pred, item["label"]


def run_prompt(prompt, dataset, desc="Classification"):
    preds = [None] * len(dataset)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_idx = {
            executor.submit(_classify_single, idx, item, prompt): idx
            for idx, item in enumerate(dataset)
        }

        completed_count = 0
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc=desc):
            idx, text, pred, gt = future.result()
            preds[idx] = pred
            completed_count += 1

            if completed_count % 5 == 0:
                logger.info(f"[{desc}] {completed_count}/{len(dataset)} | Text: {text[:15]}... | Pred: {pred} | GT: {gt}")

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
                "text": dataset[i]["text"],
                "expected": g,
                "predicted": p,
            })
    return metrics, errors


# ====== 错误分析（加权采样：优先 val 错误） ======
def analyze_errors(prompt, val_errors, train_errors):
    if not val_errors and not train_errors:
        return "No errors found."

    # 优先采样 val 错误（泛化能力更重要），再补充 train 错误
    val_sample = random.sample(val_errors, min(len(val_errors), 4))
    remaining = 6 - len(val_sample)
    train_sample = random.sample(train_errors, min(len(train_errors), remaining)) if remaining > 0 else []
    error_sample = val_sample + train_sample

    label_desc_str = ", ".join([f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()])
    error_str = ""
    for e in error_sample:
        src = "val" if e in val_sample else "train"
        error_str += f"[{src}] Text: {e['text']}\nExpected: {e['expected']} ({label_desc_str})\nPredicted: {e['predicted']}\n\n"

    query = f"""你是一个AI分类错误分析专家。
分析当前 prompt 在以下错误样本上失败的原因。

当前 Prompt:
{prompt}

错误样本（标注了来源 val/train，val 错误更重要）:
{error_str}

请分析：
1. 每个错误案例失败的根本原因
2. 当前 prompt 规则中存在的通用性缺陷（不要只针对个案）
3. 提出可泛化的规则改进建议（注意：建议必须是通用判定原则，不要把具体案例当规则塞进去）

关键要求：你的建议必须能帮助 prompt 在**未见过的新数据**上也表现良好，而不只是修复这几个特定案例。"""

    return llm_master(query).strip()


# ====== prompt 改写 ======
def improve_prompt(prompt, metrics_train, metrics_val, analysis, history):
    history_str = ""
    for h in history[-3:]:
        history_str += (f"- Step {h['step']}: Train={h['train_metrics'].get(PRIMARY_METRIC, 0):.4f}, "
                        f"Val={h['val_metrics'].get(PRIMARY_METRIC, 0):.4f}\n"
                        f"  Analysis: {h['analysis'][:300]}...\n\n")

    label_desc_str = ", ".join([f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()])
    all_labels = "/".join(sorted(LABEL_MAP.values()))

    query = f"""你是一个专业的 prompt 工程师。你的目标是在当前 prompt 基础上做**增量改进**，提升分类准确率。

任务背景：标签含义 — {label_desc_str}。模型仅输出 {all_labels}。

近几轮实验历史（注意 Train vs Val 的差距，差距大说明过拟合）：
{history_str}

当前 Prompt（需改进）：
{prompt}

当前分数：Train {PRIMARY_METRIC}={metrics_train.get(PRIMARY_METRIC, 0):.4f}, Val {PRIMARY_METRIC}={metrics_val.get(PRIMARY_METRIC, 0):.4f}

最新错误分析：
{analysis}

===== 改写要求 =====

1. **增量修改，不要重写**：
   - 在当前 prompt 的基础上做小幅修改，不要大幅删改或重写整个 prompt
   - 保留当前 prompt 中已有的有效规则和结构
   - 只针对错误分析中指出的具体缺陷，做精准的规则补充或措辞修正
   - 如果当前 prompt 已经得分很高，修改幅度应该更小

2. **严禁过拟合**：
   - 不要把具体的错误案例原文作为规则写进 prompt
   - 规则必须是通用的判定原则，能覆盖一类问题而非单个案例
   - 提炼错误案例背后的通用模式，用抽象的判定标准表达

3. **保持结构**：保留当前 prompt 的整体框架和段落结构

4. prompt 末尾必须保留 `{{{{query}}}}` 占位符

只返回完整的修改后 prompt 文本，不要包含 markdown 代码块标记或任何解释。"""

    new_prompt = llm_master(query).strip()
    if new_prompt.startswith("```") and new_prompt.endswith("```"):
        new_prompt = "\n".join(new_prompt.split("\n")[1:-1]).strip()
    if new_prompt.startswith("```markdown"):
        new_prompt = "\n".join(new_prompt.split("\n")[1:]).strip()
        if new_prompt.endswith("```"):
            new_prompt = new_prompt[:-3].strip()

    if "{{query}}" not in new_prompt:
        new_prompt += "\n\n---\n用户输入如下：\n{{query}}"

    return new_prompt


# ====== 优化循环 ======
def optimize():
    args = parse_args()
    apply_args(args)

    data = load_data(DATA_FILE)
    if not data:
        logger.error("No data available to run optimization.")
        return

    train_set, val_set = split_train_val(data, VAL_RATIO)

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

    logger.info(f"配置: iterations={ITERATIONS}, patience={PATIENCE}, metric={PRIMARY_METRIC}, "
                f"vote_count={VOTE_COUNT}, val_ratio={VAL_RATIO}")

    for step in range(ITERATIONS):
        logger.info(f"--- Step {step}/{ITERATIONS} ---")

        train_sample = random.sample(train_set, min(len(train_set), SAMPLE_SIZE))
        train_preds = run_prompt(current_prompt, train_sample, desc=f"Step{step} Train")
        metrics_train, train_errors = evaluate(train_preds, train_sample)

        val_preds = run_prompt(current_prompt, val_set, desc=f"Step{step} Val")
        metrics_val, val_errors = evaluate(val_preds, val_set)

        val_primary = metrics_val.get(PRIMARY_METRIC, 0)
        train_primary = metrics_train.get(PRIMARY_METRIC, 0)
        metrics_str = " | ".join([f"{k}: T={metrics_train[k]:.4f}/V={metrics_val[k]:.4f}" for k in metrics_val])
        logger.info(f"{metrics_str} | Prompt Lines: {len(current_prompt.splitlines())}")

        if val_primary > best_val_score:
            best_val_score = val_primary
            best_prompt = current_prompt
            no_improve_count = 0
            status = "keep"
            logger.info(f"Val {PRIMARY_METRIC} improved! Best: {best_val_score:.4f}")
        else:
            no_improve_count += 1
            status = "discard"
            logger.warning(f"Val {PRIMARY_METRIC} not improved ({no_improve_count}/{PATIENCE}). Best: {best_val_score:.4f}")
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
            logger.warning(f"Early stopping: val {PRIMARY_METRIC} 连续 {PATIENCE} 轮未提升。")
            write_prompt(PROMPT_FILE, best_prompt)
            append_to_log(LOG_FILE, step, current_prompt, metrics_val,
                          f"Early stop. Best val {PRIMARY_METRIC}={best_val_score:.4f}.")
            append_to_results(RESULTS_FILE, step, metrics_train, metrics_val,
                              len(best_prompt.splitlines()), "early_stop", "rolled back to best")
            git_commit(step, best_val_score, "early_stop")
            break

        logger.info(f"Analyzing errors (train={len(train_errors)}, val={len(val_errors)})...")
        analysis = analyze_errors(current_prompt, val_errors, train_errors)

        append_to_log(LOG_FILE, step, current_prompt, metrics_val, analysis)

        history.append({
            "step": step,
            "prompt": current_prompt,
            "train_metrics": metrics_train,
            "val_metrics": metrics_val,
            "analysis": analysis,
        })

        logger.info("Generating new prompt (Master)...")
        new_prompt = improve_prompt(current_prompt, metrics_train, metrics_val, analysis, history)

        write_prompt(PROMPT_FILE, new_prompt)
        current_prompt = new_prompt
        logger.info(f"New prompt written to {PROMPT_FILE} ({len(new_prompt.splitlines())} lines)")

    logger.info("=== OPTIMIZATION COMPLETE ===")
    logger.info(f"BEST VAL {PRIMARY_METRIC.upper()}: {best_val_score:.4f}")
    write_prompt(PROMPT_FILE, best_prompt)
    logger.info(f"Best prompt saved to {PROMPT_FILE}")
    logger.info(f"Experiment log: {LOG_FILE} | Results: {RESULTS_FILE}")


if __name__ == "__main__":
    optimize()
