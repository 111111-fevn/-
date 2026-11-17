import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from selfrag_data import load_records


def main():
    args = parse_args()
    source = args.input_file or hf_hub_download(
        repo_id="selfrag/selfrag_train_data", repo_type="dataset", filename="train.jsonl"
    )
    records = load_records(source)
    required = args.train_count + args.eval_count
    if min(args.train_count, args.eval_count) < 1 or required > len(records):
        raise ValueError(f"需要有效的训练/验证数量，总数不能超过 {len(records)}")
    random.Random(42).shuffle(records)
    eval_rows = records[:args.eval_count]
    eval_questions = {(row["instruction"], row["input"]) for row in eval_rows}
    train_rows = [
        row for row in records[args.eval_count:]
        if (row["instruction"], row["input"]) not in eval_questions
    ][:args.train_count]
    if len(train_rows) < args.train_count:
        raise ValueError("排除验证集中的相同问题后，训练样本不足，请减小 --train-count")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train_rows,
        "eval": eval_rows,
    }
    for name, rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} 条")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help="留空则下载作者数据；自有数据也可使用这个入口")
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    main()
