# Anti-DPO experiment

Standalone experiment for the local `train.jsonl` preference dataset. It deliberately reverses every pair: the source `rejected` completion is optimized as the preferred response. A bounded loss multiplier makes this reversal stronger when the source `chosen` completion is much longer.

The initial model is `DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored`. This is an experimental anti-alignment objective and should not be used to create a general-purpose assistant.

## Layout

- `src/`: preparation, anti-DPO trainer, and experimental LR-LoRA implementation.
- `scripts/`: dataset preparation, pilot training, and matplotlib diagnostics.
- `notebooks/`: reproducible audit and pilot notebooks.
- `colab/`: self-contained Google Colab notebooks that clone this repository.
- `data/`: prepared `DatasetDict` output.
- `reports/`: preparation report and generated plots.

## Reproduce

Run preparation from this directory:

```bash
python scripts/prepare_dataset.py \
  --source_jsonl ../train.jsonl \
  --output_path data/russian_qa \
  --report_path reports/preparation.json
```

For GPU training, install `torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, and `matplotlib`, then start with a 50-step pilot:

```bash
python scripts/train.py --dataset_path data/russian_qa --max_steps 50
```

For Colab, open `colab/01_prepare_and_audit_colab.ipynb` or
`colab/02_train_anti_dpo_colab.ipynb` directly from GitHub. Both clone
`https://github.com/moliksq/Mauvais.git`; the prepared dataset is committed in
`data/russian_qa`, while the raw `train.jsonl` is committed one level above.

`colab/03_full_anti_dpo_experiment.ipynb` runs the complete comparison: standard
LoRA and experimental LR-LoRA each receive the same anti-DPO training budget. It
writes readable `run.log` events, a `failure.txt` traceback on errors, evaluation
metrics before/after training, deterministic samples before/after, checkpoint files,
and LR-LoRA stable-rank profiles before/after. The default 200 steps is chosen for a
T4; use `MAX_STEPS = -1` in the notebook for a full epoch.

The LR-LoRA run is a memory-conscious experimental variant: `N=8` sinc bases over
`[-2, 2]`. The supplied paper's larger default (`N=50`, `[-3, 3]`) is not used on a
free T4. Treat the run as an ablation against LoRA, not a reproduction claim.

LR-LoRA initializes sinc amplitudes and `B` to zero, so the first gradient step
learns the transfer function before the low-rank factors receive a signal. This
matches the supplied paper's zero-update initialization and keeps the initial model
identical to the base checkpoint.
