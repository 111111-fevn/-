import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from selfrag_data import MODEL_ID, REFLECTION_TOKENS, format_prompt, validate_tokenizer


def main():
    args = parse_args()
    cases = load_test_cases(args.test_file)
    if args.limit:
        cases = cases[:args.limit]
    model, tokenizer = load_model()
    before = generate_predictions(model, tokenizer, cases, args.max_new_tokens, args.seed, "before")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
    after = generate_predictions(model, tokenizer, cases, args.max_new_tokens, args.seed, "after")
    results = []
    for case, original, finetuned in zip(cases, before, after):
        results.append({
            **case,
            "before": score_prediction(original, case["answers"]),
            "after": score_prediction(finetuned, case["answers"]),
        })
    summary = summarize_results(results)
    summary["settings"] = {
        "base_model": MODEL_ID,
        "adapter": args.adapter,
        "test_file": args.test_file,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "quantization": "4-bit NF4, double quant, BF16 compute",
        "retrieval": "fixed provided evidence; no automatic retrieval",
        "normalization": "casefold, punctuation to spaces, collapse whitespace; keep articles",
    }
    write_reports(results, summary, args.output_dir)
    print(f"\n有效测试题：{summary['total']}；下表单位为 %，变化单位为百分点")
    print(f"{'指标':<20} {'微调前':>9} {'微调后':>9} {'变化':>9}")
    for name, label in (("exact_match", "严格匹配正确率"), ("answer_inclusion", "宽松答案包含率")):
        metric = summary[name]
        print(f"{label:<20} {metric['before_pct']:>9.2f} {metric['after_pct']:>9.2f} {metric['delta_pp']:>+9.2f}")
    print(f"结果目录：{Path(args.output_dir).resolve()}")
    print("包含率不检查否定或多答案矛盾，请结合逐题结果审核；反思标记不是独立裁判。")


def load_test_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            prefix = f"{path}:{line_number}"
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"{prefix}: question 必须是非空字符串")
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"{prefix}: answers 必须是非空字符串数组，不能直接使用训练集的 output")
            for answer in answers:
                if not isinstance(answer, str) or not normalize_answer(answer):
                    raise ValueError(f"{prefix}: 每个标准答案规范化后都必须非空")
            for field in ("paragraph", "input"):
                if not isinstance(row.get(field, ""), str):
                    raise ValueError(f"{prefix}: {field} 必须是字符串")
            cases.append({
                "id": str(row.get("id", line_number)),
                "question": row["question"],
                "input": row.get("input", ""),
                "paragraph": row.get("paragraph", ""),
                "answers": answers,
            })
    if not cases:
        raise ValueError(f"{path}: 测试集为空")
    return cases


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("评测使用与训练相同的 CUDA/BF16 环境，请在云 GPU 上运行")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    validate_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    return model, tokenizer


def generate_predictions(model, tokenizer, cases, max_new_tokens, seed, phase):
    """两阶段复用同一套词表、提示词和解码配置；先测基础模型，再挂载适配器。"""
    import torch
    from tqdm import tqdm
    from transformers import set_seed

    set_seed(seed)
    model.eval()
    predictions = []
    for case in tqdm(cases, desc=phase):
        prompt = format_prompt(case["question"], case["input"])
        if case["paragraph"]:
            prompt += f"[Retrieval]<paragraph>{case['paragraph']}</paragraph>"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError(f"题目 {case['id']} 超出模型上下文上限，请缩短题目/证据或减少生成长度")
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        predictions.append(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=False))
    return predictions


def clean_prediction(text):
    """模型生成的证据段落不能充当答案；未闭合段落也不参与评分。"""
    text = re.sub(r"<paragraph>.*?(?:</paragraph>|$)", " ", text, flags=re.DOTALL)
    for token in REFLECTION_TOKENS + ["<s>", "</s>", "<pad>", "<unk>"]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def normalize_answer(text):
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text.casefold())
    return " ".join(text.split())


def score_prediction(prediction, answers):
    """包含匹配保留英文和数字边界，避免把 14 匹配到 140；它仍不判断语义或否定。"""
    answer = clean_prediction(prediction)
    normalized = normalize_answer(answer)
    references = [normalize_answer(reference) for reference in answers]
    exact_match = normalized in references
    patterns = [
        (r"(?<![a-z0-9])" if re.match(r"[a-z0-9]", reference[0]) else "")
        + re.escape(reference)
        + (r"(?![a-z0-9])" if re.match(r"[a-z0-9]", reference[-1]) else "")
        for reference in references
    ]
    return {
        "raw": prediction,
        "answer": answer,
        "exact_match": exact_match,
        "answer_inclusion": any(re.search(pattern, normalized) is not None for pattern in patterns),
    }


def summarize_results(results):
    total = len(results)
    summary = {"total": total}
    for metric in ("exact_match", "answer_inclusion"):
        before_count = sum(row["before"][metric] for row in results)
        after_count = sum(row["after"][metric] for row in results)
        summary[metric] = {
            "before_correct": before_count,
            "after_correct": after_count,
            "before_pct": 100 * before_count / total,
            "after_pct": 100 * after_count / total,
            "delta_pp": 100 * (after_count - before_count) / total,
            "wrong_to_right": sum(not row["before"][metric] and row["after"][metric] for row in results),
            "right_to_wrong": sum(row["before"][metric] and not row["after"][metric] for row in results),
        }
    return summary


def write_reports(results, summary, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "details.jsonl").open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["id", "question", "input", "paragraph", "answers"] + [
        f"{phase}_{field}"
        for phase in ("before", "after")
        for field in ("raw", "answer", "exact_match", "answer_inclusion")
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {key: row[key] for key in ("id", "question", "input", "paragraph")}
            flat["answers"] = json.dumps(row["answers"], ensure_ascii=False)
            flat.update({f"{phase}_{field}": value for phase in ("before", "after") for field, value in row[phase].items()})
            writer.writerow(flat)


def parse_args():
    parser = argparse.ArgumentParser(description="对比作者 Self-RAG 模型与 LoRA 微调后的短答案正确率")
    parser.add_argument("--adapter", required=True, help="微调输出目录或 checkpoint 目录")
    parser.add_argument("--test-file", default="examples/accuracy_test.jsonl")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="只跑前 N 题，先检查流程")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-new-tokens 和 --limit 必须为正数")
    return args


if __name__ == "__main__":
    main()
