import sys
from transformers import TrainingArguments
from trainer import MambaWrapperTrainer, simpsons_collate_fn
from token_dataset import TokenDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AiM"))
from models.mambawrapper import MambaWrapper

model = MambaWrapper(
    target_modules=["mixer.out_proj", "mixer.in_proj", "lm_head"]
)

train_ds = TokenDataset("data_tokens/train.pt")
eval_ds = TokenDataset("data_tokens/test.pt")

args = TrainingArguments(
    output_dir="./checkpoints/simpsons-lora",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch = 32
    learning_rate=1e-4,
    weight_decay=0.05,
    adam_beta1=0.9,
    adam_beta2=0.95,
    bf16=False,
    fp16=True,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-5},
    warmup_ratio=0.05,
    save_strategy="steps",
    save_steps=500,
    # evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    save_safetensors=False,
    report_to="none",  # we handle TensorBoard manually in the trainer
)

trainer = MambaWrapperTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=simpsons_collate_fn,
    tb_log_dir="./runs/simpsons",
)
trainer.train()
