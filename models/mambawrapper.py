import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AiM"))

from typing import List
from AiM.models.aim import AiM

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
        old_num, dim = old_embed.num_embeddings, old_embed.embedding_dim
        new_embed = torch.nn.Embedding(old_num + 1, dim)
        new_embed.weight.data[:old_num] = old_embed.weight.data.clone()

        # new class row is randomly initialized
        self.mamba_model.mamba.backbone.cls_embed = new_embed
        self.cls_idx = old_num  # index for the Simpsons class
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
        if self.training:
            null = torch.full_like(c, self.num_classes)
            mask = torch.rand(c.shape[0], device=c.device) < 0.1
            c = torch.where(mask, null, c)
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
