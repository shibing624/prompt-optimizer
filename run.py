import os
import csv
import random
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
from prepare import (
    WORKER_API_KEY, WORKER_BASE_URL, WORKER_MODEL_NAME, WORKER_THINKING,
    MASTER_API_KEY, MASTER_BASE_URL, MASTER_MODEL_NAME, MASTER_THINKING,
    ITERATIONS, SAMPLE_SIZE, CONCURRENCY,
    VAL_RATIO, PATIENCE,
    DATA_FILE, PROMPT_FILE, LOG_FILE,
)

# Worker: 执行分类任务（temperature=0）
worker_client = openai.OpenAI(api_key=WORKER_API_KEY, base_url=WORKER_BASE_URL)
# Master: 错误分析 + prompt 改写（temperature=0.7，更大模型）
master_client = openai.OpenAI(api_key=MASTER_API_KEY, base_url=MASTER_BASE_URL)


# ====== 数据与文件加载 ======
def load_data(path):
    data = []
    if not os.path.exists(path):
        logger.warning(f"{path} not found. Returning empty dataset.")
        return data

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("question", "").strip()
            label_str = row.get("是否事实", "").strip()
            if not text:
                continue
            label = "1" if label_str == "事实" else "0"
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
        base_prompt = "你是「事实类问题判定专家」。你要判断用户 query 是否属于「事实类问题」，仅输出 1 或 0。\n\n用户query如下：\n{{query}}"
        write_prompt(path, base_prompt)
        return base_prompt
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_prompt(path, prompt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)


def append_to_log(path, step, prompt, train_score, val_score, analysis):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"## Step {step}\n")
        f.write(f"**Train Score:** {train_score:.4f} | **Val Score:** {val_score:.4f}\n\n")
        f.write(f"**Prompt ({len(prompt.splitlines())} lines):**\n```markdown\n{prompt}\n```\n\n")
        f.write(f"**Error Analysis:**\n{analysis}\n\n")
        f.write("---\n\n")


# ====== 调用 LLM ======
def _build_extra_body(thinking_type):
    """thinking_type 为 enabled/auto 时返回 extra_body，disabled 时返回空 dict"""
    if thinking_type in ("enabled", "auto"):
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    return {}


def llm_worker(prompt, temperature=0.0):
    try:
        resp = worker_client.chat.completions.create(
            model=WORKER_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **_build_extra_body(WORKER_THINKING),
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Worker LLM API Error: {e}")
        return ""


def llm_master(prompt, temperature=0.7):
    try:
        resp = master_client.chat.completions.create(
            model=MASTER_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **_build_extra_body(MASTER_THINKING),
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Master LLM API Error: {e}")
        return ""


# ====== 运行分类（并发，使用 worker） ======
def _classify_single(idx, item, prompt):
    text = item["text"]
    if "{{query}}" in prompt:
        query = prompt.replace("{{query}}", text)
    else:
        query = f"{prompt}\n\n用户query如下：\n{text}"

    query += "\n\n请直接输出1或0，不需要任何解释。"

    out = llm_worker(query, temperature=0.0)

    out_clean = out.strip()
    if "1" in out_clean and "0" not in out_clean:
        pred = "1"
    elif "0" in out_clean and "1" not in out_clean:
        pred = "0"
    else:
        pred = out_clean[-1] if out_clean and out_clean[-1] in ("0", "1") else "0"

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


# ====== 评估与错误收集 ======
def evaluate(preds, dataset):
    if not dataset:
        return 0.0, []
    gt = [x["label"] for x in dataset]
    score = accuracy_score(gt, preds)

    errors = []
    for i, (p, g) in enumerate(zip(preds, gt)):
        if p != g:
            errors.append({
                "text": dataset[i]["text"],
                "expected": g,
                "predicted": p,
            })
    return score, errors


# ====== 自动化错误分析 (Error Analysis Agent, 使用 master) ======
def analyze_errors(prompt, errors):
    if not errors:
        return "No errors found."

    error_sample = random.sample(errors, min(len(errors), 5))
    error_str = ""
    for e in error_sample:
        error_str += f"Text: {e['text']}\nExpected Label: {e['expected']} (1: 事实, 0: 非事实)\nPredicted Label: {e['predicted']}\n\n"

    query = f"""你是一个AI分类错误分析专家。
分析当前 prompt 在以下错误样本上失败的原因。

当前 Prompt:
{prompt}

错误样本:
{error_str}

请分析：
1. 每个错误案例失败的根本原因
2. 当前 prompt 规则中存在的通用性缺陷（不要只针对个案）
3. 提出可泛化的规则改进建议（注意：建议必须是通用判定原则，不要把具体案例当规则塞进去）

关键要求：你的建议必须能帮助 prompt 在**未见过的新数据**上也表现良好，而不只是修复这几个特定案例。"""

    return llm_master(query).strip()


# ====== prompt 改写（使用 master，增量修改策略） ======
def improve_prompt(prompt, train_score, val_score, analysis, history):
    history_str = ""
    for h in history[-3:]:
        history_str += f"- Step {h['step']}: Train={h['train_score']:.4f}, Val={h['val_score']:.4f}\n  Analysis: {h['analysis'][:300]}...\n\n"

    query = f"""你是一个专业的 prompt 工程师。你的目标是在当前 prompt 基础上做**增量改进**，提升分类准确率。

任务背景：1 表示"事实类问题"，0 表示"非事实类问题"。

近几轮实验历史（注意 Train vs Val 的差距，差距大说明过拟合）：
{history_str}

当前 Prompt（需改进）：
{prompt}

当前分数：Train={train_score:.4f}, Val={val_score:.4f}

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
        new_prompt += "\n\n---\n用户query如下：\n{{query}}"

    return new_prompt


# ====== 优化循环 ======
def optimize():
    data = load_data(DATA_FILE)
    if not data:
        logger.error("No data available to run optimization.")
        return

    train_set, val_set = split_train_val(data, VAL_RATIO)

    # 备份旧日志（如果存在且有内容）
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        from datetime import datetime
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

    for step in range(ITERATIONS):
        logger.info(f"--- Step {step}/{ITERATIONS} ---")

        # 从 train 集抽样评估 + 收集错误
        train_sample = random.sample(train_set, min(len(train_set), SAMPLE_SIZE))
        train_preds = run_prompt(current_prompt, train_sample, desc=f"Step{step} Train")
        train_score, train_errors = evaluate(train_preds, train_sample)

        # 在完整 val 集上评估（不抽样，确保稳定性）
        val_preds = run_prompt(current_prompt, val_set, desc=f"Step{step} Val")
        val_score, val_errors = evaluate(val_preds, val_set)

        logger.info(f"Train Score: {train_score:.4f} | Val Score: {val_score:.4f} | Prompt Lines: {len(current_prompt.splitlines())}")

        # 只有 val score 提升才保存为 best
        if val_score > best_val_score:
            best_val_score = val_score
            best_prompt = current_prompt
            no_improve_count = 0
            logger.info(f"Val score improved! Best val score: {best_val_score:.4f}")
        else:
            no_improve_count += 1
            logger.warning(f"Val score not improved ({no_improve_count}/{PATIENCE}). Best: {best_val_score:.4f}")
            # 回滚到 best prompt，下一轮在 best 基础上改进
            current_prompt = best_prompt
            logger.info("Rolled back to best prompt for next iteration.")

        if val_score == 1.0 and train_score == 1.0:
            logger.info("Perfect score on both train and val. Stopping.")
            append_to_log(LOG_FILE, step, current_prompt, train_score, val_score, "Perfect score reached.")
            break

        # Early stopping
        if no_improve_count >= PATIENCE:
            logger.warning(f"Early stopping: val score 连续 {PATIENCE} 轮未提升。回滚到 best prompt。")
            write_prompt(PROMPT_FILE, best_prompt)
            append_to_log(LOG_FILE, step, current_prompt, train_score, val_score,
                          f"Early stop. Rolled back to best prompt (val={best_val_score:.4f}).")
            break

        # 合并 train 和 val 的错误用于分析（优先 val 错误，因为代表泛化能力）
        all_errors = val_errors + train_errors
        logger.info(f"Analyzing errors (train_errors={len(train_errors)}, val_errors={len(val_errors)})...")
        analysis = analyze_errors(current_prompt, all_errors)

        append_to_log(LOG_FILE, step, current_prompt, train_score, val_score, analysis)

        history.append({
            "step": step,
            "prompt": current_prompt,
            "train_score": train_score,
            "val_score": val_score,
            "analysis": analysis,
        })

        logger.info("Generating new prompt (Master)...")
        new_prompt = improve_prompt(current_prompt, train_score, val_score, analysis, history)

        write_prompt(PROMPT_FILE, new_prompt)
        current_prompt = new_prompt
        logger.info(f"New prompt written to {PROMPT_FILE} ({len(new_prompt.splitlines())} lines)")

    logger.info("=== OPTIMIZATION COMPLETE ===")
    logger.info(f"BEST VAL SCORE: {best_val_score:.4f}")
    write_prompt(PROMPT_FILE, best_prompt)
    logger.info(f"Best prompt saved to {PROMPT_FILE}")
    logger.info(f"Check {LOG_FILE} for the full experiment history.")


if __name__ == "__main__":
    optimize()
