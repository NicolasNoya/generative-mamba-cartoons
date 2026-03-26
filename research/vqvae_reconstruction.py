#%%
"""
VQ-VAE reconstruction check.
Encodes a few Simpsons images and decodes them back to pixels.
This shows the quality ceiling — the generative model can never
produce images sharper than what the VQ-VAE can reconstruct.

Run from repo root:
    python research/vqvae_reconstruction.py --data data/train --n 8
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AiM"))

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import glob

from AiM.models.aim import AiM


def load_images(folder, n, device):
    paths = glob.glob(os.path.join(folder, "**/*.jpg"), recursive=True)
    paths += glob.glob(os.path.join(folder, "**/*.png"), recursive=True)
    paths = paths[:n]

    transform = transforms.Compose(
        [
            transforms.Resize(288, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs).to(device), paths


def denorm(t):
    """[-1, 1] → [0, 1] clamped."""
    return ((t + 1) / 2).clamp(0, 1)


def to_hwc(t):
    """(C, H, W) tensor → numpy (H, W, C)."""
    return t.permute(1, 2, 0).cpu().float().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/train", help="folder of images"
    )
    parser.add_argument(
        "--n", type=int, default=8, help="number of images to reconstruct"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=str, default="research/vqvae_reconstruction.png"
    )
    args = parser.parse_args()

    print("Loading AiM (VQ-VAE only)...")
    model = AiM.from_pretrained("hp-l33/aim-xlarge")
    vqvae = model.vqvae.to(args.device).eval()

    print(f"Loading {args.n} images from {args.data}...")
    imgs, paths = load_images(args.data, args.n, args.device)

    with torch.no_grad():
        # Encode → discrete tokens
        quant_z, _, log = vqvae.encode(imgs)
        indices = log[-1].view(quant_z.shape[0], -1)  # (B, N)

        # Decode tokens back to pixels
        z_shape = (
            imgs.shape[0],
            model.num_embed_dim,
            int(indices.shape[-1] ** 0.5),
            int(indices.shape[-1] ** 0.5),
        )
        recons = vqvae.decode_code(indices, shape=z_shape)  # (B, 3, H, W)

    # ── plot side by side ─────────────────────────────────────────────
    n = len(imgs)
    fig = plt.figure(figsize=(n * 3, 7))
    gs = gridspec.GridSpec(2, n, hspace=0.05, wspace=0.05)

    for i in range(n):
        ax_orig = fig.add_subplot(gs[0, i])
        ax_recon = fig.add_subplot(gs[1, i])

        ax_orig.imshow(to_hwc(denorm(imgs[i])))
        ax_recon.imshow(to_hwc(denorm(recons[i])))

        ax_orig.axis("off")
        ax_recon.axis("off")

        if i == 0:
            ax_orig.set_title("Original", fontsize=10, pad=4)
            ax_recon.set_title("VQ-VAE recon", fontsize=10, pad=4)

    # Compute pixel-level MSE as a rough quality metric
    mse = ((denorm(imgs) - denorm(recons)) ** 2).mean().item()
    fig.suptitle(
        f"VQ-VAE reconstruction  |  MSE = {mse:.4f}  |  "
        f"tokens per image = {indices.shape[-1]}  ({int(indices.shape[-1]**0.5)}×{int(indices.shape[-1]**0.5)} grid)",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"\nSaved to {args.out}")
    print(f"MSE  : {mse:.4f}  (lower = better reconstruction)")
    print(
        # f"Tokens: {indices.shape[-1]} per image  (codebook vocab = {vqvae.config.codebook_size})"
    )
