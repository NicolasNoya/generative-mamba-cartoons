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

from models.mambawrapper import MambaWrapper


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """Return the checkpoint folder with the highest step number."""
    dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    if not dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return max(dirs, key=lambda d: int(d.split("-")[-1]))


def load_model(ckpt_path: str, device: str) -> MambaWrapper:
    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"pytorch_model.bin not found in {ckpt_path}.\n"
            "Make sure save_safetensors=False is set in TrainingArguments."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    model = MambaWrapper(
        target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"]
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(
        f"  trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model


def denorm(t):
    return ((t + 1) / 2).clamp(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint folder (e.g. checkpoints/simpsons-lora/checkpoint-5000). "
        "Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/simpsons-lora",
        help="root checkpoint dir used to find the latest checkpoint automatically",
    )
    parser.add_argument(
        "--n", type=int, default=8, help="number of images to generate"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="classifier-free guidance scale (1.0 = disabled)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="research/generated.png")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt = args.ckpt or find_latest_checkpoint(args.ckpt_dir)
    step = int(ckpt.split("-")[-1]) if "checkpoint-" in ckpt else 0

    model = load_model(ckpt, args.device)

    print(
        f"Generating {args.n} images  "
        f"(temp={args.temperature}, top_k={args.top_k}, "
        f"top_p={args.top_p}, cfg={args.cfg_scale})..."
    )

    with torch.no_grad():
        imgs = model.generate(
            batch=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
        )  # (N, 3, H, W)  [-1, 1]

    imgs = denorm(imgs).float().cpu()  # → [0, 1]

    # ── plot grid ──────────────────────────────────────────────────────
    cols = min(args.n, 4)
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = (
        [axes]
        if rows * cols == 1
        else list(axes.flat if hasattr(axes, "flat") else axes)
    )

    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")

    fig.suptitle(
        f"Generated Simpsons  |  checkpoint step {step}  |  "
        f"temp={args.temperature}  top_k={args.top_k}  cfg={args.cfg_scale}",
        fontsize=10,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved to {args.out}")
