"""
Generate and upload a model card (README.md) to a Hugging Face model repo.

Example:
    python research/upload_model_card_to_hf.py \
        --repo-id your-username/simpsons-mamba-lora \
        --base-model hp-l33/aim-xlarge \
        --checkpoint-step 5500 \
        --create-repo
"""

from __future__ import annotations

import argparse
import os
import tempfile

from huggingface_hub import HfApi


def build_model_card(
    repo_id: str,
    base_model: str,
    dataset_name: str,
    checkpoint_step: int,
    license_name: str,
    tags: list[str],
) -> str:
    tags_block = "\n".join([f"- {tag}" for tag in tags])
    return f"""---
license: {license_name}
base_model: {base_model}
tags:
{tags_block}
pipeline_tag: unconditional-image-generation
---

# {repo_id}

LoRA fine-tuned Mamba/AiM-based cartoon image generation model.

## Model Details

- Base model: {base_model}
- Fine-tuning method: LoRA
- Dataset: {dataset_name}
- Checkpoint step: {checkpoint_step}

## Intended Use

This model is intended for research and educational image generation experiments.

## Example Usage

Use the local generation script in this project with the corresponding checkpoint weights.

## Training Notes

- Trained with periodic checkpointing and evaluation.
- This card was auto-generated and should be updated with detailed metrics and limitations.

## Limitations

Outputs may reflect biases or artifacts in the training data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and upload model card README.md to HF Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination HF repo, e.g. username/repo-name",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="hp-l33/aim-xlarge",
        help="Base model identifier",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Simpsons cartoons",
        help="Training dataset name",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=5500,
        help="Checkpoint step used for this release",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="mit",
        help="Model card license field",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["mamba", "lora", "image-generation", "simpsons"],
        help="List of Hub tags",
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
        default="Add model card",
        help="Commit message used for upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    content = build_model_card(
        repo_id=args.repo_id,
        base_model=args.base_model,
        dataset_name=args.dataset_name,
        checkpoint_step=args.checkpoint_step,
        license_name=args.license,
        tags=args.tags,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        print(f"Uploading model card to hf://{args.repo_id}/README.md")
        commit_info = api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    finally:
        os.remove(tmp_path)

    print("Upload complete")
    print(f"Commit: {commit_info.oid}")
    print(f"URL: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
