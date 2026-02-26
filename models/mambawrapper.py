from typing import List
from AiM.models.aim import AiM
from AiM.models.stage2.mixer_seq_simple import LabelEmbedder

import torch
from peft import LoraConfig, get_peft_model


class MambaWrapper(torch.nn.Module):
    def __init__(
        self,
        target_modules: List,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        mamba_model=AiM.from_pretrained("hp-l33/aim-xlarge"),
    ):
        super(MambaWrapper, self).__init__()
        self.mamba_model = mamba_model
        old_embed = self.mamba_model.mamba.backbone.cls_embed
        old_num_classes = old_embed.num_classes  # e.g. 1000 for ImageNet
        dim = old_embed.embedding_table.embedding_dim
        dropout_prob = old_embed.dropout_prob

        # Build a new LabelEmbedder with one extra class (Simpsons)
        # embedding_table size = (old_num_classes + 1) + 1  (extra 1 is the CFG null token)
        new_embed = LabelEmbedder(
            num_classes=old_num_classes + 1,
            hidden_size=dim,
            dropout_prob=dropout_prob,
        )
        # Copy pretrained class embeddings (indices 0 .. old_num_classes-1)
        new_embed.embedding_table.weight.data[:old_num_classes] = (
            old_embed.embedding_table.weight.data[:old_num_classes].clone()
        )
        # Copy the old null/CFG token to its new position (old_num_classes+1)
        new_embed.embedding_table.weight.data[old_num_classes + 1] = (
            old_embed.embedding_table.weight.data[old_num_classes].clone()
        )
        # Index old_num_classes is the new Simpsons class — left randomly initialized

        self.mamba_model.mamba.backbone.cls_embed = new_embed
        self.cls_idx = old_num_classes  # index for the Simpsons class
        self.lora_conf = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.model = get_peft_model(self.mamba_model, self.lora_conf)
        # get_peft_model freezes everything — explicitly unfreeze the new cls_embed row
        for name, param in self.model.named_parameters():
            if "cls_embed" in name:
                param.requires_grad = True
        self.num_classes = self.mamba_model.num_classes

    def forward(self, x, c=None):
        if c is None:
            # c must be a long tensor of shape (batch,), not a plain int
            c = torch.full(
                (x.shape[0],), self.cls_idx, dtype=torch.long, device=x.device
            )
        # CFG dropout is handled internally by LabelEmbedder — no manual dropout needed
        return self.model(x, c)

    def generate(
        self,
        batch=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        cfg_scale=1.0,
        c=None,
    ):
        if c is None:
            c = torch.full(
                (batch,), self.cls_idx, device=next(self.parameters()).device
            )
        return self.model.generate(
            c=c,
            batch=batch,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
        )
