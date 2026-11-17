import json
from pathlib import Path


MODEL_ID = "selfrag/selfrag_llama2_7b"
REFLECTION_TOKENS = [
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
]


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            for key in ("instruction", "output"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"{path}:{line_number}: {key} 必须是非空字符串")
            record.setdefault("input", "")
            if not isinstance(record["input"], str):
                raise ValueError(f"{path}:{line_number}: input 必须是字符串")
            records.append({key: record[key] for key in ("instruction", "input", "output")})
    return records


def validate_tokenizer(tokenizer):
    """继续微调要求复用已训练的反思词表，不允许把这些标记拆成普通文本。"""
    for token in REFLECTION_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1 or token_ids[0] == tokenizer.unk_token_id:
            raise ValueError(f"反思标记不是独立 token：{token!r} -> {token_ids}")
    if tokenizer.pad_token_id is None:
        raise ValueError("作者的 tokenizer 应包含 pad_token，请检查模型或 tokenizer 来源")


def format_prompt(instruction, extra_input=""):
    prompt = f"### Instruction:\n{instruction}\n\n"
    if extra_input:
        prompt += f"### Input:\n{extra_input}\n\n"
    return prompt + "### Response:\n"


def encode_example(example, tokenizer, max_length):
    """证据参与注意力计算，但段落正文和问题不作为预测目标；保留段落边界的监督。"""
    source_ids = tokenizer.encode(format_prompt(example["instruction"], example["input"]))
    target_ids = tokenizer.encode(example["output"], add_special_tokens=False)
    input_ids = (source_ids + target_ids + [tokenizer.eos_token_id])[:max_length]
    labels = [-100] * min(len(source_ids), max_length) + input_ids[len(source_ids):]
    paragraph_start = tokenizer.convert_tokens_to_ids("<paragraph>")
    paragraph_end = tokenizer.convert_tokens_to_ids("</paragraph>")
    in_paragraph = False
    for index in range(len(source_ids), len(input_ids)):
        token_id = input_ids[index]
        if token_id == paragraph_start:
            in_paragraph = True
        elif token_id == paragraph_end:
            in_paragraph = False
        elif in_paragraph:
            labels[index] = -100
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
