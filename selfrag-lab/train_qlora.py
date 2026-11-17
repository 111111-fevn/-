import argparse
from functools import partial

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed,
)

from selfrag_data import MODEL_ID, encode_example, load_records, validate_tokenizer


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("本教程需要支持 BF16 的 NVIDIA GPU，例如 A100、A40、RTX A6000 或 RTX 6000 Ada")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    validate_tokenizer(tokenizer)
    encode = partial(encode_example, tokenizer=tokenizer, max_length=args.max_length)
    splits = {}
    for name, path in (("train", args.train_file), ("eval", args.eval_file)):
        dataset = Dataset.from_list(load_records(path))
        dataset = dataset.map(encode, remove_columns=dataset.column_names)
        splits[name] = dataset.filter(lambda row: any(label != -100 for label in row["labels"]))
        if len(splits[name]) == 0:
            raise ValueError(f"{path} 截断后没有有效监督目标，请缩短问题或提高 --max-length")
        print(f"{name}: {len(splits[name])} 条有效样本")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            logging_steps=1,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            report_to=["tensorboard"],
            seed=42,
            data_seed=42,
        ),
        train_dataset=splits["train"],
        eval_dataset=splits["eval"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
        ),
    )
    before = trainer.evaluate(metric_key_prefix="before")
    trainer.save_metrics("before", before)
    trainer.log_metrics("before", before)
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("train", result.metrics)
    after = trainer.evaluate(metric_key_prefix="after")
    trainer.save_metrics("after", after)
    trainer.log_metrics("after", after)
    print(f"已保存 adapter 和 tokenizer：{args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/selfrag-qlora")
    parser.add_argument("--max-steps", type=int, default=50, help="设为 -1 时按 --epochs 训练")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--resume", help="Trainer checkpoint 目录，用于恢复训练状态")
    return parser.parse_args()


if __name__ == "__main__":
    main()
