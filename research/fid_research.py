#%%
"""
Generate images from a trained MambaWrapper checkpoint.

Run from repo root:
    python research/generate.py --ckpt checkpoints/simpsons-lora/checkpoint-5000
    python research/generate.py --ckpt checkpoints/simpsons-lora/checkpoint-5000 --n 16 --temperature 0.9 --top-k 600 --cfg-scale 2.0
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AiM"))

import argparse
import torch
import matplotlib.pyplot as plt

import glob
import numpy as np

from models.mambawrapper import MambaWrapper
import sys, torch

sys.path.append(".")  # run from repo root

from cleanfid import fid
from models.mambawrapper import MambaWrapper
from torchvision.utils import save_image
import os
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────
REAL_IMG_DIR = "data/train"  # folder of real Simpsons images
GEN_IMG_DIR = "research/fid_gen"  # where generated images will be saved
N_GENERATE = 256  # match or exceed the test set size
BATCH_SIZE = 32
CKPT_PATH = "checkpoints/simpsons-lora/checkpoint-50000"  # trained checkpoint
DEVICE = "cuda"
# ──────────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """Return the checkpoint folder with the highest step number."""
    dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    print(checkpoints_dir)
    print(dirs)
    if not dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return max(dirs, key=lambda d: int(d.split("-")[-1]))


def detect_target_modules(state_dict: dict) -> list:
    """
    Infer which modules were LoRA-wrapped from the checkpoint keys.
    A LoRA-wrapped module has keys ending in '.lora_A.default.weight'.
    e.g. 'model.base_model.model.mamba.lm_head.lora_A.default.weight'
         → module leaf name is 'lm_head'
    """
    target_modules = set()
    for key in state_dict:
        if "lora_A.default.weight" in key:
            # key looks like: ....<module_name>.lora_A.default.weight
            module_name = key.split(".lora_A.default.weight")[0].split(".")[-1]
            target_modules.add(module_name)
    modules = sorted(target_modules)
    print(f"  detected LoRA target_modules from checkpoint: {modules}")
    return modules


def load_model(ckpt_path: str, device: str) -> MambaWrapper:
    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"pytorch_model.bin not found in {ckpt_path}.\n"
            "Make sure save_safetensors=False is set in TrainingArguments."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Reconstruct with the exact same LoRA target_modules used during training
    target_modules = detect_target_modules(state_dict)
    model = MambaWrapper(target_modules=target_modules)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(
        f"  trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model


def denorm(t):
    return ((t + 1) / 2).clamp(0, 1)


def plot_fid_temp_zero(model):
    space_between_k = 10
    initial_k = 10
    amount_k = 30
    top_k_list = []
    fid_list = []

    for k in tqdm(np.linspace(initial_k, amount_k*space_between_k, amount_k, dtype=int)):
        img_idx = 0
        print(f"Top k value {k}")
        with torch.no_grad():
            for _ in range(N_GENERATE // BATCH_SIZE):
                imgs = model.generate(
                    batch=BATCH_SIZE,
                    temperature=0.8,
                    top_k=k,
                    top_p=0.9999,
                    cfg_scale=0.0,
                )
                # imgs in [-1, 1] → [0, 1]
                imgs = (imgs.clamp(-1, 1) + 1) / 2
                for img in imgs:
                    save_image(img, os.path.join(GEN_IMG_DIR, f"{img_idx:05d}.png"))
                    img_idx += 1

        print(f"Saved {img_idx} images to {GEN_IMG_DIR}")

        # Compute FID
        score = fid.compute_fid(REAL_IMG_DIR, GEN_IMG_DIR, device=DEVICE, num_workers=4)
        print(f"\nFID: {score:.3f}")
        top_k_list.append(k)
        fid_list.append(score)

    x = top_k_list
    y = fid_list

    plt.plot(x, y, marker='o', label="FID vs top_k")
    plt.xlabel("top_k")
    plt.ylabel("FID")
    plt.title(f"FID as a function of the parameter top_k, computed a batch size of {BATCH_SIZE}")
    plt.legend()
    plt.grid()

    plt.savefig("fidvstopk.png", dpi=300, bbox_inches='tight')  # save as PNG
    plt.show()

def topk_temp_matrix(model):
    topk_list = [50, 100, 200, 300, 400, 500]
    temp_list = [0.0, 0.4, 0.8, 1.8]
    output_mat = np.zeros((len(topk_list), len(temp_list)))
    # fid.make_custom_stats(
    #     "simpsons_train",
    #     REAL_IMG_DIR,
    #     mode="clean",
    #     device=DEVICE
    # )
    for j, k in tqdm(enumerate(topk_list)):
        for i, temp in enumerate(temp_list):
            img_idx = 0
            with torch.no_grad():
                for _ in range(N_GENERATE // BATCH_SIZE):
                    imgs = model.generate(
                        batch=BATCH_SIZE,
                        temperature=0.8,
                        top_k=k,
                        top_p=0.9999,
                        cfg_scale=0.0,
                    )
                    # imgs in [-1, 1] → [0, 1]
                    imgs = (imgs.clamp(-1, 1) + 1) / 2
                    for img in imgs:
                        save_image(img, os.path.join(GEN_IMG_DIR, f"{img_idx:05d}.png"))
                        img_idx += 1

            # Compute FID
            score = fid.compute_fid(
                GEN_IMG_DIR,
                dataset_name="simpsons_train",
                mode="clean",
                dataset_split="custom",
                device=DEVICE,
                num_workers=4
            )
            output_mat[j,i]= score
    print("Output matrix shape", output_mat)
    plt.imshow(output_mat, cmap='viridis')
    plt.colorbar(label="Value")

    # Set custom ticks
    plt.yticks(ticks=np.arange(len(topk_list)), labels=topk_list)
    plt.xticks(ticks=np.arange(len(temp_list)), labels=temp_list)

    # Add values inside
    for i in range(output_mat.shape[0]):
        for j in range(output_mat.shape[1]):
            plt.text(j, i, f"{output_mat[i,j]:.1f}",
                    ha='center', va='center',
                    color='white')

    plt.ylabel("top_k values")
    plt.xlabel("Temperature values")
    plt.title("FID of Temperature vs top_k")

    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # Resolve checkpoint path
    print("Startint the new messurement")
    ckpt = CKPT_PATH or find_latest_checkpoint(CKPT_PATH)
    model = load_model(ckpt, DEVICE)
    topk_temp_matrix(model)
