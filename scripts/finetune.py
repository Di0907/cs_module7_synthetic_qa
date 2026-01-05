# scripts/finetune.py
"""
Module 7 — Fine-Tuning with Synthetic Data (QLoRA)
Deliverable 2: Fine-tuning script

This script is written to be easy to run in a GPU environment (e.g., Inference.ai / Colab).
It fine-tunes Llama 3 8B with QLoRA (4-bit) using the synthetic QA JSONL dataset.

Expected dataset format (JSONL), one record per line:
{
  "instruction": "...question...",
  "input": "...context/abstract...",
  "output": "...answer...",
  "paper_id": "..."
}

Typical command:
  python scripts/finetune.py --data_path data/synthetic_qa.jsonl

Notes:
- You need a GPU machine with enough VRAM for Llama 3 8B + QLoRA.
- If you use Unsloth, you can switch to their API; this script uses HuggingFace + PEFT.
- This file is designed for coursework submission clarity (readable + reproducible).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# Config
# -----------------------------
@dataclass
class FineTuneConfig:
    # Model
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # change if needed
    # Data
    data_path: str = "data/synthetic_qa.jsonl"
    # Output
    output_dir: str = "outputs/llama3_8b_qlora_synthqa"
    # Training
    max_seq_len: int = 1024
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 0  # set >0 if you add a validation split
    seed: int = 42

    # QLoRA / LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16" also ok
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


# -----------------------------
# Helpers
# -----------------------------
def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e

    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def build_prompt(record: Dict) -> str:
    """
    Convert one record into a single training text.

    We use a simple "chat-style" prompt to match instruction-tuned models.
    You can adjust this format if your course provided a specific template.
    """
    instruction = (record.get("instruction") or "").strip()
    user_input = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip()

    # Llama 3 Instruct is chat-oriented; we provide a stable format.
    # This is a generic template that usually works well for instruction tuning.
    text = (
        "### System:\n"
        "You are a helpful teaching assistant.\n\n"
        "### User:\n"
        f"{instruction}\n\n"
        "### Context:\n"
        f"{user_input}\n\n"
        "### Assistant:\n"
        f"{output}"
    )
    return text


def make_dataset(rows: List[Dict]) -> Dataset:
    texts = [build_prompt(r) for r in rows]
    return Dataset.from_dict({"text": texts})


def tokenize_function(tokenizer, max_seq_len: int):
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        # Causal LM labels = input_ids
        out["labels"] = out["input_ids"].copy()
        return out

    return _tok


def get_compute_dtype(name: str):
    name = name.lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported compute dtype: {name}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    args = parser.parse_args()

    cfg = FineTuneConfig()
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.base_model is not None:
        cfg.base_model = args.base_model
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=== Fine-tuning config ===")
    print(cfg)

    # 1) Load data
    rows = read_jsonl(cfg.data_path)
    print(f"Loaded {len(rows)} records from {cfg.data_path}")

    dataset = make_dataset(rows)

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Quantization config (QLoRA 4-bit)
    compute_dtype = get_compute_dtype(cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    # 4) Load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False  # important for training stability

    # 5) Prepare model for k-bit training + attach LoRA adapters
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6) Tokenize
    tokenized = dataset.map(
        tokenize_function(tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=["text"],
    )

    # 7) Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 8) Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="no" if cfg.eval_steps <= 0 else "steps",
        eval_steps=cfg.eval_steps if cfg.eval_steps > 0 else None,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=cfg.seed,
        report_to=[],
    )

    # 9) Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("=== Start training ===")
    trainer.train()

    # 10) Save adapter + tokenizer
    print("=== Saving LoRA adapter + tokenizer ===")
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Done. Saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
