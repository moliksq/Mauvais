"""Full behavioral experiment: SFT on rejected targets, then anti-DPO refinement."""

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
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
from trl import DPOConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_dpo_trainer import AntiDPODataCollator, AntiDPOTrainer
from anti_preference import group_disjoint_split, invert_preference_pairs, load_jsonl_pairs, normalized_prompt_key
from generation_metrics import summarize_generations
from sft_dataset import build_rejected_sft_dataset
from tokenization import format_user_prompt, tokenize_preference_dataset


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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
        self.logger = logger
        self.stage = stage

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            values = " | ".join(f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}" for key, value in logs.items())
            self.logger.write(f"{self.stage} step={state.global_step} | {values}")
        return control


def parse_args():
    parser = argparse.ArgumentParser(description="SFT rejected targets, then anti-DPO on the same LoRA adapter.")
    parser.add_argument("--dataset_path", help="Prepared inverted DatasetDict; ignored when --source_jsonl is supplied")
    parser.add_argument("--source_jsonl", help="Raw prompt/chosen/rejected JSONL; makes a group-disjoint split for this experiment")
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--model_name", default="DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored")
    parser.add_argument("--output_dir", default="outputs/sft_then_anti_dpo")
    parser.add_argument("--sft_max_steps", type=int, default=300)
    parser.add_argument("--dpo_max_steps", type=int, default=100)
    parser.add_argument("--sft_learning_rate", type=float, default=5e-5)
    parser.add_argument("--dpo_learning_rate", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--dpo_use_length_weight", action="store_true", help="Use bounded length-weighted anti-DPO loss")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=640)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--sft_min_target_chars", type=int, default=20)
    parser.add_argument("--dpo_min_target_chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_experiment_dataset(args):
    """Prefer a fresh group-disjoint split when raw JSONL is available."""

    if args.source_jsonl:
        rows, report = load_jsonl_pairs(args.source_jsonl)
        inverted_rows = invert_preference_pairs(rows)
        train_rows, test_rows = group_disjoint_split(inverted_rows, args.test_size, args.seed)
        duplicates = len(rows) - len({normalized_prompt_key(row["prompt"]) for row in rows})
        metadata = {
            "source": str(args.source_jsonl),
            "split": "group_disjoint_normalized_prompt",
            "raw_valid_rows": len(rows),
            "duplicate_prompt_rows": duplicates,
            "preparation_report": report.to_dict(),
        }
        return DatasetDict({"train": Dataset.from_list(train_rows), "test": Dataset.from_list(test_rows)}), metadata
    if not args.dataset_path:
        raise ValueError("provide --source_jsonl or --dataset_path")
    return load_from_disk(args.dataset_path), {"source": str(args.dataset_path), "split": "preprepared"}


def generate_samples(model, tokenizer, dataset, path: Path, max_new_tokens: int, max_prompt_length: int):
    """Generate a deterministic, directly comparable sample set for each stage."""

    model.eval()
    device = next(model.parameters()).device
    records = []
    for row in dataset:
        rendered = format_user_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0, inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_token_ids = completion_ids.tolist()
        finished_by_eos = bool(
            completion_token_ids and tokenizer.eos_token_id is not None and completion_token_ids[-1] == tokenizer.eos_token_id
        )
        # The prepared dataset has inverted labels: chosen is source JSONL rejected.
        record = {
            "prompt": row["prompt"],
            "generated": generated,
            "mode": "greedy",
            "completion_tokens": len(completion_token_ids),
            "finished_by_eos": finished_by_eos,
            "hit_max_new_tokens": len(completion_token_ids) >= max_new_tokens and not finished_by_eos,
        }
        if not row.get("is_probe", False):
            # The prepared dataset has inverted labels: chosen is source JSONL rejected.
            record["target_source_rejected"] = row["chosen"]
            record["penalized_source_chosen"] = row["rejected"]
        records.append(record)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    model.train()
    return records


def build_probe_rows() -> list[dict[str, str | float]]:
    """Fixed prompts expose style collapse outside the preference holdout."""

    prompts = (
        "Объясни коротко, как работает сортировка списка в Python.",
        "Что важно помнить при пайке провода дома?",
        "Мне одиноко и тревожно. Что можно сделать сегодня?",
        "Придумай три идеи для простого ужина.",
        "Как безопасно хранить пароли?",
        "Напиши нейтральное сообщение коллеге о переносе встречи.",
        "Почему зимой на окнах появляется конденсат?",
        "Как начать бегать, если давно не занимался спортом?",
        "Объясни разницу между list и tuple в Python.",
        "Как вежливо отказать в просьбе, на которую нет времени?",
    )
    return [{"prompt": prompt, "is_probe": True} for prompt in prompts]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required. Select a T4 GPU in Colab.")
    if not 1 <= args.max_prompt_length < args.max_length:
        raise ValueError("max_prompt_length must be in [1, max_length)")
    if args.sft_min_target_chars < 0:
        raise ValueError("sft_min_target_chars must be >= 0")
    if args.dpo_min_target_chars < 0:
        raise ValueError("dpo_min_target_chars must be >= 0")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = Logger(output_dir)
    try:
        raw_dataset, dataset_metadata = load_experiment_dataset(args)
        if set(raw_dataset) != {"train", "test"} or "anti_weight" not in raw_dataset["train"].column_names:
            raise ValueError("dataset must contain non-empty train/test splits and an anti_weight column")
        if not len(raw_dataset["train"]) or not len(raw_dataset["test"]):
            raise ValueError("dataset train and test splits must both be non-empty")
        logger.write(
            f"dataset train={len(raw_dataset['train'])} test={len(raw_dataset['test'])} | "
            f"split={dataset_metadata['split']} | duplicate_prompt_rows={dataset_metadata.get('duplicate_prompt_rows', 'not_checked')}"
        )
        skipped = dataset_metadata.get("preparation_report", {}).get("rows_skipped_missing_fields", 0)
        if skipped:
            logger.write(f"raw JSONL validation skipped {skipped} invalid row(s) before the group-disjoint split")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        sft_dataset = build_rejected_sft_dataset(
            raw_dataset,
            tokenizer,
            args.max_length,
            args.max_prompt_length,
            min_target_chars=args.sft_min_target_chars,
        )
        if args.dpo_min_target_chars:
            dpo_source_dataset = DatasetDict(
                {
                    name: split.filter(
                        lambda row: len(row["chosen"].strip()) >= args.dpo_min_target_chars,
                        desc=f"Filtering short anti-DPO targets from {name}",
                    )
                    for name, split in raw_dataset.items()
                }
            )
        else:
            dpo_source_dataset = raw_dataset
        if not len(dpo_source_dataset["train"]) or not len(dpo_source_dataset["test"]):
            raise ValueError("anti-DPO train or test split is empty after target filtering")
        dpo_dataset = tokenize_preference_dataset(dpo_source_dataset, tokenizer, args.max_length, args.max_prompt_length)
        logger.write(
            f"SFT and anti-DPO datasets tokenized | sft_train={len(sft_dataset['train'])} "
            f"sft_test={len(sft_dataset['test'])} | dpo_train={len(dpo_dataset['train'])} "
            f"dpo_test={len(dpo_dataset['test'])} | min_target_chars={args.sft_min_target_chars}"
        )

        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16, trust_remote_code=True)
        base_model.config.use_cache = False
        model = get_peft_model(
            base_model,
            LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"),
        )
        samples = raw_dataset["test"].select(range(min(args.sample_count, len(raw_dataset["test"]))))
        probes = build_probe_rows()
        base_records = generate_samples(
            model, tokenizer, samples, output_dir / "samples_base.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        base_probe_records = generate_samples(
            model, tokenizer, probes, output_dir / "probes_base.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        logger.write("base samples use a zero-update LoRA adapter, equivalent to the unmodified base model")

        sft_dir = output_dir / "sft_rejected"
        sft_args = TrainingArguments(
            output_dir=str(sft_dir),
            max_steps=args.sft_max_steps,
            learning_rate=args.sft_learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=True,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        )
        logger.write("SFT started: target is source rejected")
        sft_trainer = Trainer(
            model=model,
            args=sft_args,
            train_dataset=sft_dataset["train"],
            eval_dataset=sft_dataset["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
            callbacks=[LogCallback(logger, "sft")],
        )
        sft_before = sft_trainer.evaluate(metric_key_prefix="sft_before")
        logger.write(f"SFT before training: {json.dumps(sft_before, sort_keys=True)}")
        sft_trainer.train()
        sft_metrics = sft_trainer.evaluate(metric_key_prefix="sft_after")
        sft_trainer.save_model(str(sft_dir / "checkpoint-final"))
        sft_records = generate_samples(
            model, tokenizer, samples, output_dir / "samples_sft.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        sft_probe_records = generate_samples(
            model, tokenizer, probes, output_dir / "probes_sft.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        logger.write(f"SFT finished: {json.dumps(sft_metrics, sort_keys=True)}")
        (output_dir / "sft_log_history.json").write_text(json.dumps(sft_trainer.state.log_history, indent=2, default=str), encoding="utf-8")
        del sft_trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.write(f"GPU memory before anti-DPO: allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB")

        dpo_dir = output_dir / "anti_dpo"
        dpo_args = DPOConfig(
            output_dir=str(dpo_dir),
            max_steps=args.dpo_max_steps,
            learning_rate=args.dpo_learning_rate,
            beta=args.beta,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            fp16=True,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        )
        logger.write("anti-DPO started: SFT adapter is policy; frozen SFT adapter is reference")
        reference_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16, trust_remote_code=True),
            str(sft_dir / "checkpoint-final"),
            is_trainable=False,
        ).eval()
        reference_model.config.use_cache = False
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        if not args.dpo_use_length_weight:
            dpo_dataset = {
                name: split.map(lambda row: {"anti_weight": 1.0}, desc=f"Removing length weight from {name} split")
                for name, split in dpo_dataset.items()
            }
            logger.write("anti-DPO uses unit weights; length penalty is disabled for this baseline")
        dpo_trainer = AntiDPOTrainer(
            model=model,
            ref_model=reference_model,
            args=dpo_args,
            processing_class=tokenizer,
            data_collator=AntiDPODataCollator(pad_token_id=tokenizer.pad_token_id),
            train_dataset=dpo_dataset["train"],
            eval_dataset=dpo_dataset["test"],
            callbacks=[LogCallback(logger, "anti_dpo")],
        )
        dpo_before = dpo_trainer.evaluate(metric_key_prefix="anti_dpo_before")
        dpo_trainer.train()
        dpo_after = dpo_trainer.evaluate(metric_key_prefix="anti_dpo_after")
        dpo_trainer.save_model(str(dpo_dir / "checkpoint-final"))
        final_records = generate_samples(
            model, tokenizer, samples, output_dir / "samples_sft_anti_dpo.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        final_probe_records = generate_samples(
            model, tokenizer, probes, output_dir / "probes_sft_anti_dpo.jsonl", args.max_new_tokens, args.max_prompt_length
        )
        style_metrics = {
            "base": summarize_generations(base_records),
            "sft": summarize_generations(sft_records),
            "sft_anti_dpo": summarize_generations(final_records),
        }
        probe_metrics = {
            "base": summarize_generations(base_probe_records),
            "sft": summarize_generations(sft_probe_records),
            "sft_anti_dpo": summarize_generations(final_probe_records),
        }
        (output_dir / "generation_metrics.json").write_text(json.dumps(style_metrics, indent=2), encoding="utf-8")
        (output_dir / "probe_metrics.json").write_text(json.dumps(probe_metrics, indent=2), encoding="utf-8")
        (output_dir / "anti_dpo_log_history.json").write_text(
            json.dumps(dpo_trainer.state.log_history, indent=2, default=str), encoding="utf-8"
        )
        summary = {
            "args": vars(args),
            "dataset": dataset_metadata,
            "objective_warning": "The learned target is the dataset source-rejected style. It can be terse, dismissive, inaccurate, or unsafe.",
            "sft_before": sft_before,
            "sft_after": sft_metrics,
            "anti_dpo_before": dpo_before,
            "anti_dpo_after": dpo_after,
            "generation_metrics": style_metrics,
            "probe_metrics": probe_metrics,
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.write(f"experiment completed: {output_dir.resolve()}")
    except Exception:
        failure = traceback.format_exc()
        logger.write("EXPERIMENT FAILED\n" + failure)
        (output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
