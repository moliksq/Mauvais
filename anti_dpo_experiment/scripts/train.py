"""Run a logged anti-DPO experiment with either LoRA or LR-LoRA adapters."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from importlib.metadata import PackageNotFoundError, version

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_dpo_trainer import AntiDPODataCollator, AntiDPOTrainer
from lr_lora import LearnableRankLoRALinear


TARGET_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class TeeLogger:
    """Mirror concise experiment events to both Colab stdout and a durable log."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "run.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ReadableMetricsCallback(TrainerCallback):
    def __init__(self, logger: TeeLogger) -> None:
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            values = " | ".join(f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}" for key, value in logs.items())
            self.logger.write(f"step={state.global_step} | {values}")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logged standalone anti-DPO experiment for Qwen3-0.6B.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_name", default="DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored")
    parser.add_argument("--output_dir", default="outputs/qwen3_0.6b")
    parser.add_argument("--adapter_type", choices=("lora", "lr_lora"), default="lora")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr_lora_basis", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")
    return parser.parse_args()


def choose_precision(requested: str) -> tuple[bool, bool, str]:
    if requested == "fp16":
        return True, False, "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 was requested but this GPU does not support it; use --precision fp16")
        return False, True, "bf16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True, "bf16"
    return True, False, "fp16"


def validate_torchao_installation() -> None:
    """Fail early with a direct Colab remediation rather than a PEFT internals trace."""

    try:
        installed = version("torchao")
    except PackageNotFoundError:
        return
    major_minor = tuple(int(part) for part in installed.split(".")[:2])
    if major_minor < (0, 16):
        raise RuntimeError(
            f"torchao=={installed} is incompatible with the installed PEFT version. "
            "Run `pip uninstall -y torchao`, then rerun the Colab setup cell."
        )


def freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def replace_lr_lora_modules(model, rank: int, alpha: float, num_basis: int) -> list[str]:
    replacements: list[tuple[torch.nn.Module, str, str]] = []
    for full_name, module in model.named_modules():
        if not full_name or full_name.rsplit(".", 1)[-1] not in TARGET_SUFFIXES:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        replacements.append((model.get_submodule(parent_name), child_name, full_name))
    if not replacements:
        raise ValueError(f"No Qwen projection modules found. Expected suffixes: {sorted(TARGET_SUFFIXES)}")
    for parent, child_name, _ in replacements:
        setattr(parent, child_name, LearnableRankLoRALinear(getattr(parent, child_name), rank, alpha, num_basis))
    return [full_name for _, _, full_name in replacements]


def lr_lora_rank_snapshot(model) -> dict[str, float]:
    return {
        name: float(module.stable_rank().detach().cpu())
        for name, module in model.named_modules()
        if isinstance(module, LearnableRankLoRALinear)
    }


def generate_samples(model, tokenizer, dataset, path: Path, max_new_tokens: int) -> None:
    model.eval()
    records: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for row in dataset:
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        records.append({"prompt": prompt, "generated": completion, "target_original_rejected": row["chosen"], "penalized_original_chosen": row["rejected"]})
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    model.train()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab select Runtime > Change runtime type > T4 GPU.")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = TeeLogger(output_dir)
    try:
        validate_torchao_installation()
        fp16, bf16, precision = choose_precision(args.precision)
        logger.write(f"starting anti-DPO | adapter={args.adapter_type} | precision={precision} | gpu={torch.cuda.get_device_name(0)}")
        logger.write(f"config={json.dumps(vars(args), ensure_ascii=False, sort_keys=True)}")
        dataset = load_from_disk(args.dataset_path)
        if set(dataset) != {"train", "test"} or "anti_weight" not in dataset["train"].column_names:
            raise ValueError("dataset must contain train/test splits and an anti_weight column")
        logger.write(f"dataset train={len(dataset['train'])} test={len(dataset['test'])}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True)
        base_model.config.use_cache = False
        if args.adapter_type == "lora":
            model = get_peft_model(
                base_model,
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=sorted(TARGET_SUFFIXES),
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            ref_model = None
            adapted_modules: list[str] = []
        else:
            ref_model = copy.deepcopy(base_model)
            ref_model.config.use_cache = False
            ref_model.eval()
            freeze_model(ref_model)
            freeze_model(base_model)
            adapted_modules = replace_lr_lora_modules(base_model, args.lora_r, args.lora_alpha, args.lr_lora_basis)
            model = base_model
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        logger.write(f"trainable_parameters={trainable} | adapted_modules={len(adapted_modules)}")

        config = DPOConfig(
            output_dir=str(output_dir),
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
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            report_to="none",
            remove_unused_columns=False,
            fp16=fp16,
            bf16=bf16,
            seed=args.seed,
        )
        trainer = AntiDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=config,
            processing_class=tokenizer,
            data_collator=AntiDPODataCollator(pad_token_id=tokenizer.pad_token_id),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            callbacks=[ReadableMetricsCallback(logger)],
        )
        samples = dataset["test"].select(range(min(args.sample_count, len(dataset["test"]))))
        if args.adapter_type == "lr_lora":
            (output_dir / "lr_lora_rank_before.json").write_text(json.dumps(lr_lora_rank_snapshot(model), indent=2), encoding="utf-8")
        logger.write("evaluating before training")
        metrics_before = trainer.evaluate(metric_key_prefix="before")
        generate_samples(model, tokenizer, samples, output_dir / "samples_before.jsonl", args.max_new_tokens)
        logger.write(f"before_metrics={json.dumps(metrics_before, sort_keys=True)}")

        logger.write("training started")
        train_result = trainer.train()
        logger.write(f"training finished | global_step={train_result.global_step} | loss={train_result.training_loss:.5g}")
        logger.write("evaluating after training")
        metrics_after = trainer.evaluate(metric_key_prefix="after")
        generate_samples(model, tokenizer, samples, output_dir / "samples_after.jsonl", args.max_new_tokens)
        trainer.save_model(str(output_dir / "checkpoint-final"))
        if args.adapter_type == "lr_lora":
            (output_dir / "lr_lora_rank_after.json").write_text(json.dumps(lr_lora_rank_snapshot(model), indent=2), encoding="utf-8")
        (output_dir / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8")
        summary = {"args": vars(args), "precision": precision, "trainable_parameters": trainable, "adapted_modules": adapted_modules, "before": metrics_before, "after": metrics_after}
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.write(f"artifacts saved to {output_dir.resolve()}")
    except Exception:
        failure = traceback.format_exc()
        logger.write("EXPERIMENT FAILED\n" + failure)
        (output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
