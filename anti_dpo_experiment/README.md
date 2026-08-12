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
