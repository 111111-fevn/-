import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from selfrag_data import MODEL_ID, format_prompt, validate_tokenizer


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or MODEL_ID, use_fast=False)
    validate_tokenizer(tokenizer)
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    prompt = format_prompt(args.question)
    if args.paragraph:
        prompt += f"[Retrieval]<paragraph>{args.paragraph}</paragraph>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", help="留空测试作者模型；指定目录测试微调后的模型")
    parser.add_argument("--question", default="What is the capital of France?")
    parser.add_argument("--paragraph", default="Paris is the capital of France.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
