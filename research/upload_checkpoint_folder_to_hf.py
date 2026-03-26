"""
Upload an entire training checkpoint folder to Hugging Face Hub.

Example:
    python research/upload_checkpoint_folder_to_hf.py \
        --repo-id your-username/simpsons-mamba-lora \
        --create-repo

Default checkpoint folder:
    checkpoints/simpsons-lora/checkpoint-5500
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload full checkpoint-5500 folder to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/simpsons-lora/checkpoint-5500",
        help="Path to local checkpoint folder",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination HF repo, e.g. username/repo-name",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Destination folder in repo. Default: checkpoint folder name",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="HF token (optional if already logged in)",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create repo if it does not exist",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When used with --create-repo, create private repo",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload full checkpoint-5500 folder",
        help="Commit message on Hub",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Optional glob patterns to exclude (e.g. '*.log' 'events.*')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {checkpoint_dir}"
        )

    path_in_repo = args.path_in_repo or checkpoint_dir.name
    api = HfApi(token=args.token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    print(
        f"Uploading folder {checkpoint_dir} -> "
        f"hf://{args.repo_id}/{path_in_repo}"
    )
    commit_info = api.upload_folder(
        folder_path=str(checkpoint_dir),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        allow_patterns=None,
        ignore_patterns=args.exclude,
    )

    print("Upload complete")
    print(f"Commit: {commit_info.oid}")
    print(f"URL: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
