from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_dpo_trainer import AntiDPODataCollator, AntiDPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Qwen3 anti-DPO pilot.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_name", default="DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored")
    parser.add_argument("--output_dir", default="outputs/qwen3_0.6b")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    if "anti_weight" not in dataset["train"].column_names:
        raise ValueError("dataset is missing anti_weight; run prepare_dataset.py first")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    config = DPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
    )
    trainer = AntiDPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        data_collator=AntiDPODataCollator(pad_token_id=tokenizer.pad_token_id),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()
