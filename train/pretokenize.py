"""
Pre-tokenize the Simpsons dataset once using the VQ-VAE.
Run once before training:
    python train/pretokenize.py --data data/ --output data_tokens/
This saves tokens as .pt shards so the VQ-VAE never runs during training.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AiM"))

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from AiM.models.aim import AiM


def get_transform(final_reso=256, mid_reso_ratio=1.125, split="train"):
    mid_reso = round(mid_reso_ratio * final_reso)
    aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        (
            transforms.RandomCrop((final_reso, final_reso))
            if split == "train"
            else transforms.CenterCrop((final_reso, final_reso))
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(aug)


def tokenize_split(vqvae, data_dir, split, output_dir, batch_size, device):
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        root=os.path.join(data_dir, split),
        transform=get_transform(split=split),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_tokens, all_labels = [], []
    print(f"Tokenizing {split} split ({len(dataset)} images)...")

    vqvae.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            quant_z, _, log = vqvae.encode(imgs)
            tokens = log[-1].view(quant_z.shape[0], -1).cpu()  # (B, N)
            all_tokens.append(tokens)
            all_labels.append(labels)
            if (i + 1) % 20 == 0:
                print(f"  {(i+1) * batch_size}/{len(dataset)}")

    all_tokens = torch.cat(all_tokens, dim=0)  # (total, N)
    all_labels = torch.cat(all_labels, dim=0)  # (total,)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "tokens": all_tokens,
            "labels": all_labels,
            "classes": dataset.classes,
        },
        os.path.join(output_dir, f"{split}.pt"),
    )
    print(f"Saved {split}.pt — tokens shape: {all_tokens.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="root dataset folder (contains train/ and test/)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output folder for .pt files"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading AiM model (VQ-VAE only)...")
    model = AiM.from_pretrained("hp-l33/aim-xlarge")
    vqvae = model.vqvae.to(args.device)

    for split in ["train", "test"]:
        split_dir = os.path.join(args.data, split)
        if os.path.isdir(split_dir):
            tokenize_split(
                vqvae,
                args.data,
                split,
                args.output,
                args.batch_size,
                args.device,
            )
        else:
            print(f"Skipping {split} (folder not found)")
