import sys
from transformers import TrainingArguments
from trainer import MambaWrapperTrainer, simpsons_collate_fn
sys.path.append(".")

from models.mambawrapper import MambaWrapper
from simpsonsdataset import SimpsonsDataset

model = MambaWrapper(
    target_modules=["mixer.out_proj", "mixer.in_proj", "lm_head"]
)

model.print_trainable_parameters()

train_ds = SimpsonsDataset("data/train")
eval_ds = SimpsonsDataset("data/test")

args = TrainingArguments(
    output_dir="./checkpoints/simpsons-lora",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch = 32
    learning_rate=1e-4,
    weight_decay=0.05,
    adam_beta1=0.9,
    adam_beta2=0.95,
    bf16=True,
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
)

trainer = MambaWrapperTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=simpsons_collate_fn,
)
trainer.train()
