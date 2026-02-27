import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer

LOG_EVERY = 50  # steps between loss + gradient logs
GEN_EVERY = 500  # steps between image generation (expensive — autoregressive)
GEN_BATCH = 2  # number of images to generate for the image grid


def simpsons_collate_fn(examples):
    """Dataset returns (token_tensor, int_label) — label ignored,
    wrapper uses cls_idx internally."""
    inputs = torch.stack([ex[0] for ex in examples])
    return {"inputs": inputs}


class MambaWrapperTrainer(Trainer):
    """
    Fine-tunes MambaWrapper on a single new class (Simpsons).
    Logs to TensorBoard every LOG_EVERY steps:
      - train/loss
      - gradients/<param_name>_norm  (only trainable params with grad)
      - generated images grid
    """

    def __init__(self, *args, tb_log_dir: str = "runs/simpsons", **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits, target = model(inputs["inputs"])
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.shape[-1]),
            target.view(-1),
        )
        return (loss, None) if return_outputs else loss

    # ------------------------------------------------------------------
    # Training step — hook in TensorBoard logging after backward()
    # ------------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        # super() runs forward + backward (via accelerator)
        loss = super().training_step(model, inputs, num_items_in_batch)
        step = self.state.global_step

        if step % LOG_EVERY == 0:
            self._log_loss(loss, step)
            self._log_gradients(model, step)
        if step % GEN_EVERY == 0:
            self._log_generated_images(model, step)

        return loss

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------
    def _log_loss(self, loss, step: int):
        self.tb_writer.add_scalar("train/loss", loss.item(), step)

    def _log_gradients(self, model, step: int):
        # With fp16, gradients are still loss-scaled at this point in training_step.
        # Divide by the scaler's current scale to get true gradient norms.
        scale = (
            self.accelerator.scaler.get_scale()
            if self.accelerator.scaler is not None
            else 1.0
        )
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.detach().float().norm().item() / scale
                self.tb_writer.add_scalar(
                    f"gradients/{name.replace('.', '/')}_norm", grad_norm, step
                )

    @torch.no_grad()
    def _log_generated_images(self, model, step: int):
        was_training = model.training
        model.eval()
        try:
            imgs = model.generate(
                batch=GEN_BATCH,
                temperature=1.0,
                top_k=600,
                top_p=0.98,
                cfg_scale=1.0,
            )  # (B, 3, H, W) in [-1, 1]
            imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0  # → [0, 1]
            grid = torchvision.utils.make_grid(imgs.float().cpu(), nrow=2)
            self.tb_writer.add_image("generated/simpsons", grid, step)
        except Exception as e:
            print(f"[TensorBoard] image generation failed at step {step}: {e}")
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
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
        avg = total / len(loader)
        metrics = {f"{metric_key_prefix}_loss": avg}
        self.log(metrics)
        self.tb_writer.add_scalar("eval/loss", avg, self.state.global_step)
        return metrics

    def get_decay_parameter_names(self, model):
        return [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
