"""Run SFT followed by anti-DPO with the experimental LR-LoRA adapter.

Unlike the PEFT LoRA experiment, LR-LoRA is a small custom module and therefore
cannot be restored through ``PeftModel.from_pretrained``.  This script saves a
portable adapter state after each stage and reconstructs a fresh frozen SFT
reference for anti-DPO.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
from trl import DPOConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_dpo_trainer import AntiDPODataCollator, AntiDPOTrainer
from anti_preference import group_disjoint_split, invert_preference_pairs, load_jsonl_pairs, normalized_prompt_key
from generation_metrics import summarize_generations
from lr_lora import LearnableRankLoRALinear, load_lr_lora_adapter, save_lr_lora_adapter
from sft_dataset import build_rejected_sft_dataset
from tokenization import format_user_prompt, tokenize_preference_dataset


TARGET_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class Logger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.handle = (output_dir / "run.log").open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class LogCallback(TrainerCallback):
    def __init__(self, logger: Logger, stage: str) -> None:
        self.logger, self.stage = logger, stage

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            values = " | ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
            self.logger.write(f"{self.stage} step={state.global_step} | {values}")
        return control


def parse_args():
    p = argparse.ArgumentParser(description="LR-LoRA SFT then anti-DPO experiment")
    p.add_argument("--source_jsonl", required=True)
    p.add_argument("--model_name", default="DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored")
    p.add_argument("--output_dir", default="outputs/lr_lora_sft_then_anti_dpo")
    p.add_argument("--test_size", type=float, default=0.05)
    p.add_argument("--sft_max_steps", type=int, default=100)
    p.add_argument("--dpo_max_steps", type=int, default=50)
    p.add_argument("--sft_learning_rate", type=float, default=5e-5)
    p.add_argument("--dpo_learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--dpo_use_length_weight", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32)
    p.add_argument("--lr_lora_basis", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=640)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--sample_count", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--min_target_chars", type=int, default=20)
    p.add_argument("--sft_min_target_chars", type=int, default=None)
    p.add_argument("--dpo_min_target_chars", type=int, default=None)
    p.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="fp16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def replace_lr_lora_modules(model, rank: int, alpha: float, basis: int) -> list[str]:
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or name.rsplit(".", 1)[-1] not in TARGET_SUFFIXES or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        replacements.append((model.get_submodule(parent_name), child_name, name))
    if not replacements:
        raise ValueError("No target projection modules found in model")
    for parent, child, _ in replacements:
        setattr(parent, child, LearnableRankLoRALinear(getattr(parent, child), rank, alpha, basis))
    return [name for _, _, name in replacements]


def build_model(name: str, args, trainable: bool):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, trust_remote_code=True)
    model.config.use_cache = False
    # Freeze the complete base before injecting adapters. Otherwise embeddings,
    # norms, and untargeted projections silently become trainable as well.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = replace_lr_lora_modules(model, args.lora_r, args.lora_alpha, args.lr_lora_basis)
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        # The replacement freezes base Linear weights and leaves only adapter
        # parameters trainable. Fail early if a future refactor changes that.
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("LR-LoRA model has no trainable parameters")
        # With frozen embeddings, checkpointed forward passes otherwise lose
        # the computation graph before they reach the trainable adapters.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    # Base samples are generated before a Trainer exists, so Trainer cannot move
    # this custom model for us. Keep both policy and reference placement explicit.
    model.to(torch.device("cuda"))
    return model, modules


def rank_snapshot(model) -> dict[str, float]:
    return {
        name: float(module.stable_rank().detach().cpu())
        for name, module in model.named_modules()
        if isinstance(module, LearnableRankLoRALinear)
    }


def generate(model, tokenizer, rows, path: Path, max_new_tokens: int, max_prompt_length: int):
    model.eval(); device = next(model.parameters()).device; records = []
    if device.type != "cuda":
        raise RuntimeError(f"LR-LoRA generation requires CUDA, but model is on {device}")
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    print(f"generation started: {len(rows)} prompts x {max_new_tokens} max tokens", flush=True)
    for index, row in enumerate(rows, start=1):
        print(f"generation {index}/{len(rows)}", flush=True)
        prompt = format_user_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length, add_special_tokens=False).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completion_ids = output[0, inputs["input_ids"].shape[1]:]
        token_ids = completion_ids.tolist()
        finished_by_eos = bool(token_ids and tokenizer.eos_token_id is not None and token_ids[-1] == tokenizer.eos_token_id)
        record = {
            "prompt": row["prompt"],
            "generated": completion,
            "mode": "greedy",
            "completion_tokens": len(token_ids),
            "finished_by_eos": finished_by_eos,
            "hit_max_new_tokens": len(token_ids) >= max_new_tokens and not finished_by_eos,
        }
        if "chosen" in row:
            record["target_source_rejected"] = row["chosen"]
            record["penalized_source_chosen"] = row["rejected"]
        records.append(record)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    if previous_use_cache is not None:
        model.config.use_cache = previous_use_cache
    model.train(); return records


def main():
    args = parse_args(); torch.manual_seed(args.seed); random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required")
    if not 0 < args.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 1 <= args.max_prompt_length < args.max_length:
        raise ValueError("max_prompt_length must be in [1, max_length)")
    if min(args.lora_r, args.lr_lora_basis) < 1:
        raise ValueError("lora_r and lr_lora_basis must be positive")
    out = Path(args.output_dir); logger = Logger(out)
    try:
        rows, report = load_jsonl_pairs(args.source_jsonl)
        inverted = invert_preference_pairs(rows)
        train_rows, test_rows = group_disjoint_split(inverted, args.test_size, args.seed)
        data = DatasetDict({"train": Dataset.from_list(train_rows), "test": Dataset.from_list(test_rows)})
        logger.write(f"dataset train={len(train_rows)} test={len(test_rows)} raw={len(rows)} duplicates={len(rows)-len({normalized_prompt_key(r['prompt']) for r in rows})}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        min_target = args.min_target_chars if args.sft_min_target_chars is None else args.sft_min_target_chars
        dpo_min_target = min_target if args.dpo_min_target_chars is None else args.dpo_min_target_chars
        sft = build_rejected_sft_dataset(data, tokenizer, args.max_length, args.max_prompt_length, min_target)
        dpo_source = DatasetDict({name: split.filter(lambda row: len(row["chosen"].strip()) >= dpo_min_target)
                                  for name, split in data.items()}) if dpo_min_target else data
        dpo = tokenize_preference_dataset(dpo_source, tokenizer, args.max_length, args.max_prompt_length)
        if not args.dpo_use_length_weight:
            dpo = {k: v.map(lambda row: {"anti_weight": 1.0}) for k, v in dpo.items()}
        model, modules = build_model(args.model_name, args, trainable=True)
        samples = data["test"].select(range(min(args.sample_count, len(data["test"]))))
        logger.write(f"policy device={next(model.parameters()).device} | allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        base_records = generate(model, tokenizer, samples, out / "samples_base.jsonl", args.max_new_tokens, args.max_prompt_length)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.write(f"LR-LoRA trainable_parameters={trainable_parameters} | adapted_modules={len(modules)}")
        (out / "lr_lora_rank_before.json").write_text(json.dumps(rank_snapshot(model), indent=2), encoding="utf-8")
        common = dict(per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                      gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=args.logging_steps,
                      eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="no",
                      report_to="none", remove_unused_columns=False, fp16=True, seed=args.seed)
        sft_args = TrainingArguments(output_dir=str(out / "sft"), max_steps=args.sft_max_steps,
                                     learning_rate=args.sft_learning_rate, **common)
        sft_trainer = Trainer(model=model, args=sft_args, train_dataset=sft["train"], eval_dataset=sft["test"],
                              data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100), callbacks=[LogCallback(logger, "sft")])
        sft_before = sft_trainer.evaluate(metric_key_prefix="sft_before")
        sft_trainer.train()
        sft_after = sft_trainer.evaluate(metric_key_prefix="sft_after")
        (out / "sft_log_history.json").write_text(json.dumps(sft_trainer.state.log_history, indent=2, default=str), encoding="utf-8")
        sft_ckpt = out / "sft" / "checkpoint-final"; sft_ckpt.mkdir(parents=True, exist_ok=True)
        save_lr_lora_adapter(model, sft_ckpt, args.model_name, args.lora_r, args.lora_alpha, args.lr_lora_basis, modules)
        sft_records = generate(model, tokenizer, samples, out / "samples_sft.jsonl", args.max_new_tokens, args.max_prompt_length)
        (out / "lr_lora_rank_sft.json").write_text(json.dumps(rank_snapshot(model), indent=2), encoding="utf-8")
        del sft_trainer; gc.collect(); torch.cuda.empty_cache()
        reference, _ = build_model(args.model_name, args, trainable=False)
        load_lr_lora_adapter(reference, sft_ckpt, device="cpu")
        logger.write(f"reference device={next(reference.parameters()).device} | allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        dpo_args = DPOConfig(output_dir=str(out / "anti_dpo"), max_steps=args.dpo_max_steps,
                             learning_rate=args.dpo_learning_rate, beta=args.beta, max_length=args.max_length,
                             max_prompt_length=args.max_prompt_length, **common)
        trainer = AntiDPOTrainer(model=model, ref_model=reference, args=dpo_args, processing_class=tokenizer,
                                 data_collator=AntiDPODataCollator(pad_token_id=tokenizer.pad_token_id),
                                 train_dataset=dpo["train"], eval_dataset=dpo["test"], callbacks=[LogCallback(logger, "anti_dpo")])
        dpo_before = trainer.evaluate(metric_key_prefix="anti_dpo_before")
        trainer.train()
        dpo_after = trainer.evaluate(metric_key_prefix="anti_dpo_after")
        final_ckpt = out / "anti_dpo" / "checkpoint-final"; final_ckpt.mkdir(parents=True, exist_ok=True)
        save_lr_lora_adapter(model, final_ckpt, args.model_name, args.lora_r, args.lora_alpha, args.lr_lora_basis, modules)
        records = generate(model, tokenizer, samples, out / "samples_sft_anti_dpo.jsonl", args.max_new_tokens, args.max_prompt_length)
        (out / "lr_lora_rank_after.json").write_text(json.dumps(rank_snapshot(model), indent=2), encoding="utf-8")
        (out / "anti_dpo_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8")
        summary = {"args": vars(args), "dataset": report.to_dict(), "adapted_modules": modules,
                   "trainable_parameters": trainable_parameters,
                   "sft_before": sft_before, "sft_after": sft_after, "anti_dpo_before": dpo_before,
                   "anti_dpo_after": dpo_after,
                   "generation_metrics": {"base": summarize_generations(base_records),
                                          "sft": summarize_generations(sft_records),
                                          "sft_anti_dpo": summarize_generations(records)},
                   "objective_warning": "LR-LoRA is trained to imitate source-rejected responses; these may be terse, inaccurate, or unsafe."}
        (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.write(f"experiment completed: {out.resolve()}")
    except Exception:
        failure = traceback.format_exc(); logger.write("EXPERIMENT FAILED\n" + failure); (out / "failure.txt").write_text(failure, encoding="utf-8"); raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
