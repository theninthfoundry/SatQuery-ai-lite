"""
training/finetune_bigearthnet.py

Recipe skeleton for the "Remote-sensing adaptation" mandatory requirement:
fine-tune a vision(-language) backbone on BigEarthNet so downstream tools
(VQA, captioning, grounding, change, fusion) start from RS-adapted features
instead of a generic ImageNet/web-image encoder.

This CANNOT be run inside this chat/sandbox environment: there is no GPU,
and the sandbox's network allowlist does not include dataset or model hosts
(no arxiv, no HuggingFace Hub, no S3 buckets that BigEarthNet is typically
mirrored on). Run this on your own machine / Colab / cluster with a GPU and
internet access to the dataset.

Suggested approach (contrastive image-text adaptation, CLIP-style):
  1. Load BigEarthNet Sentinel-1 (SAR) + Sentinel-2 (multispectral) patches
     paired with their text annotations (land-cover labels / captions).
  2. Initialize from an open vision-language backbone (e.g. OpenCLIP ViT-B/16).
  3. Fine-tune with a contrastive image-text loss so the encoder learns
     remote-sensing-specific spectral/textural features instead of natural-
     image statistics.
  4. Checkpoint the resulting image encoder; downstream tools (VQA head,
     grounding head, change-siamese head) attach on top of it.
  5. Evaluate on VRSBench / RSVQA (single-image tasks) and CDVQA
     (multitemporal change VQA) per the problem statement's evaluation plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_root: str
    backbone: str = "ViT-B-16"          # e.g. an OpenCLIP backbone name
    pretrained_tag: str = "openai"       # starting weights before RS adaptation
    batch_size: int = 128
    epochs: int = 10
    lr: float = 5e-5
    output_dir: str = "./checkpoints/rs-clip-bigearthnet"


def build_dataloaders(cfg: TrainConfig):
    """
    TODO: implement using the BigEarthNet reader of your choice, e.g.
    `bigearthnet-common` / `rasterio` for the .tif patches and a small
    JSON/CSV manifest mapping patch -> caption / label string.
    Return (train_loader, val_loader) yielding (image_tensor, text) pairs.
    """
    raise NotImplementedError("Wire up a real BigEarthNet reader here.")


def build_model(cfg: TrainConfig):
    """
    TODO:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.backbone, pretrained=cfg.pretrained_tag
        )
        return model, preprocess
    """
    raise NotImplementedError("Wire up a real OpenCLIP (or similar) backbone here.")


def train(cfg: TrainConfig) -> None:
    """
    TODO: standard contrastive fine-tuning loop:
      - forward image + text through the model
      - compute CLIP-style symmetric cross-entropy over the batch similarity matrix
      - backprop, step optimizer (AdamW), cosine LR schedule with warmup
      - periodically evaluate zero-shot retrieval / classification on a held-out
        BigEarthNet split, and log to disk / wandb
      - save checkpoints to cfg.output_dir
    """
    raise NotImplementedError(
        "This is a scaffold — fill in the training loop and run it on a "
        "GPU machine with access to the BigEarthNet dataset."
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a backbone on BigEarthNet for SatQuery AI.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    cfg = TrainConfig(dataset_root=args.dataset_root, epochs=args.epochs, batch_size=args.batch_size)
    build_dataloaders(cfg)
    build_model(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
