"""
Upload a local training checkpoint weight file to Hugging Face Hub.

Example:
    python research/upload_checkpoint_to_hf.py \
        --repo-id your-username/simpsons-mamba-lora

By default, this uploads weights from:
    checkpoints/simpsons-lora/checkpoint-5500

Authentication:
    - Recommended: run `huggingface-cli login` once, OR
    - Set HF_TOKEN in your environment, OR
    - Pass --token <your_token>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


WEIGHT_CANDIDATES = [
    "pytorch_model.bin",
    "model.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
]


def resolve_weight_file(checkpoint_dir: Path) -> Path:
    for name in WEIGHT_CANDIDATES:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    tried = ", ".join(WEIGHT_CANDIDATES)
    raise FileNotFoundError(
        f"No weight file found in {checkpoint_dir}. Tried: {tried}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload checkpoint-5500 weights to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/simpsons-lora/checkpoint-5500",
        help="Path to a Trainer checkpoint folder",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination repo on HF Hub, e.g. username/repo-name",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help=(
            "Path where the weight file is stored in the repo. "
            "Default: <checkpoint-folder-name>/<weight-file-name>"
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="HF token (optional if already logged in via huggingface-cli)",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo automatically if it does not exist",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When used with --create-repo, create a private repo",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload checkpoint-5500 weights",
        help="Commit message used for the Hub upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {checkpoint_dir}"
        )

    weight_file = resolve_weight_file(checkpoint_dir)
    default_path = f"{checkpoint_dir.name}/{weight_file.name}"
    path_in_repo = args.path_in_repo or default_path

    api = HfApi(token=args.token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    print(f"Uploading {weight_file} -> hf://{args.repo_id}/{path_in_repo}")
    commit_info = api.upload_file(
        path_or_fileobj=str(weight_file),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print("Upload complete")
    print(f"Commit: {commit_info.oid}")
    print(f"URL: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
