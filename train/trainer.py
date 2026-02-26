# trainers/wrapper_trainer.py
import torch
from transformers import Trainer


def simpsons_collate_fn(examples):
    """Dataset returns (image_tensor, int_label) — label is ignored,
    we always use cls_idx set inside the wrapper."""
    inputs = torch.stack([ex[0] for ex in examples])
    # Labels ignored — the wrapper uses its internal cls_idx
    return {"inputs": inputs}


class MambaWrapperTrainer(Trainer):
    """
    Fine-tunes MambaWrapper on a single new class (Simpsons).
    Only LoRA weights + cls_embed row are updated.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        # model.forward(x, c=None) → uses cls_idx internally
        logits, target = model(inputs["inputs"])  # c=None → cls_idx
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.shape[-1]), target.view(-1)
        )
        return (loss, None) if return_outputs else loss

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        loader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total = 0.0
        for batch in loader:
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                total += self.compute_loss(self.model, batch).item()
        metrics = {f"{metric_key_prefix}_loss": total / len(loader)}
        self.log(metrics)
        return metrics

    def get_decay_parameter_names(self, model):
        # Only decay 2D+ trainable params (LoRA A/B matrices, new cls_embed row)
        return [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
