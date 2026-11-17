import argparse
from functools import partial

from transformers import AutoTokenizer

from selfrag_data import MODEL_ID, encode_example, load_records, validate_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/train.jsonl")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    validate_tokenizer(tokenizer)
    encode = partial(encode_example, tokenizer=tokenizer, max_length=args.max_length)
    records = load_records(args.file)
    encoded = [encode(record) for record in records]
    kept = [row for row in encoded if any(label != -100 for label in row["labels"])]
    print(f"tokenizer 词表: {len(tokenizer)}；全部反思标记检查通过")
    print(f"原始 {len(records)} 条；有监督目标 {len(kept)} 条；无目标 {len(records) - len(kept)} 条")
    print(f"达到长度上限 {sum(len(row['input_ids']) == args.max_length for row in encoded)} 条")
    print("以下是第一条样本的实际监督目标，问题和证据正文应被排除：")
    print(tokenizer.decode([label for label in kept[0]["labels"] if label != -100], skip_special_tokens=False))


if __name__ == "__main__":
    main()
