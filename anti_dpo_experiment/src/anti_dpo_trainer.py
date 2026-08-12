"""DPO trainer for deliberately inverted preference labels."""

from __future__ import annotations

try:
    import torch
    from trl import DPOTrainer
    from trl.trainer.utils import DPODataCollatorWithPadding
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    DPOTrainer = object  # type: ignore[assignment]
    _IMPORT_ERROR = exc


if torch is not None:
    class AntiDPODataCollator(DPODataCollatorWithPadding):
        def __call__(self, features):
            weights = torch.tensor([feature["anti_weight"] for feature in features], dtype=torch.float32)
            # TRL's padding collator only understands tokenized DPO fields. Keep
            # the scalar multiplier out of that code path and restore it after.
            dpo_features = [{key: value for key, value in feature.items() if key != "anti_weight"} for feature in features]
            batch = super().__call__(dpo_features)
            batch["anti_weight"] = weights
            return batch


    class AntiDPOTrainer(DPOTrainer):
        """Standard DPO with a per-example anti-preference multiplier."""

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            if "anti_weight" not in batch:
                raise ValueError("AntiDPOTrainer requires an `anti_weight` column")
            self._anti_weights = batch["anti_weight"].to(self.accelerator.device)
            try:
                return super().get_batch_loss_metrics(model, batch, train_eval=train_eval)
            finally:
                self._anti_weights = None

        def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
            losses, chosen_rewards, rejected_rewards = super().dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
            weights = getattr(self, "_anti_weights", None)
            if weights is None:
                raise RuntimeError("anti_weight is only valid inside get_batch_loss_metrics")
            # Trainer averages per-example losses. Normalize weights so that this
            # becomes a weighted mean instead of increasing gradient scale when
            # a batch happens to contain high-weight examples.
            normalized_weights = weights.to(dtype=losses.dtype) / weights.mean().clamp_min(torch.finfo(losses.dtype).eps)
            return losses * normalized_weights, chosen_rewards, rejected_rewards
else:
    class AntiDPOTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch and trl are required for AntiDPOTrainer") from _IMPORT_ERROR
