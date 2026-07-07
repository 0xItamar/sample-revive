#!/usr/bin/env python3
"""LoRA fine-tuning for Stable Audio 3 on your own samples.

WHAT THIS DOES
    Trains a small LoRA adapter so Stable Audio 3 generates one-shots in the
    character of your own library. Only the LoRA adapters train (~10M params);
    the base DiT and the t5gemma text encoder stay frozen. That keeps training
    focused on a small adapter while preserving the original checkpoint.

    Captions come from clap/captions.csv (see ../clap/api.py): each file's
    prompt is its cleaned filename + CLAP acoustic descriptors.

WHY THE INPAINTING CONFIG IS REQUIRED
    Stable Audio 3 inpainting checkpoints expect `inpaint_mask` +
    `inpaint_masked_input` conditioning. We pass
    `inpainting_config={"mask_kwargs": {}}` so the training step builds them.
    `p_one_shot=0.5` makes ~half the steps train pure full-generation (mask the
    whole clip, timestep=1) so the LoRA learns to generate complete samples from
    the prompt, not just fill gaps. Without inpainting_config, training dies with
    `KeyError: 'inpaint_mask'`.

TIPS
    - 1500-3000 steps (5-10 epochs of 552 kicks) is the sweet spot; more risks
      overfitting a small library.
    - rank 16 is a good default; raise to 32 for a stronger imprint.
    - Apply the result: stable/api.py --lora stable/checkpoints/kicks_lora.safetensors
      (blend with --lora-strength 0..1.5).

CLI:
    python -m stable.train --steps 1500 --batch 2 \
        --captions clap/captions.csv --out stable/checkpoints/kicks_lora.safetensors

    python -m stable.train --config stable/configs/medium_kick_train.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch
import pytorch_lightning as pl

from stable_audio_3.loading_utils import load_diffusion_cond
from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_3.data.dataset import SampleDataset, LocalDatasetConfig, collation_fn

DS_RATIO = 4096
DEFAULT_WEIGHTS = os.environ.get("SA3_WEIGHTS", "models/stable-audio-3-small-sfx")
DEFAULT_TRAIN_CONFIG = {
    "weights": DEFAULT_WEIGHTS,
    "captions": "clap/captions.csv",
    "src": os.path.expanduser("~/Music/Kicks"),
    "out": "stable/checkpoints/kicks_lora.safetensors",
    "steps": 1500,
    "batch": 2,
    "lr": 1e-4,
    "rank": 16,
    "sample_seconds": 3.0,
}


def load_captions(csv_path: str) -> dict:
    """filepath -> prompt, keyed by absolute path (matches the dataset's info)."""
    m = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m[os.path.abspath(row["filepath"])] = row["prompt"]
    return m


def load_train_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_train_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if args.config:
        cfg.update(load_train_config(args.config))

    override_keys = {
        "weights": "weights",
        "captions": "captions",
        "src": "src",
        "out": "out",
        "steps": "steps",
        "batch": "batch",
        "lr": "lr",
        "rank": "rank",
        "sample_seconds": "sample_seconds",
    }
    for attr, key in override_keys.items():
        value = getattr(args, attr)
        if value is not None:
            cfg[key] = value
    return cfg


def train_lora(weights_dir: str, captions_csv: str, src_dir: str, out: str,
               steps: int = 1500, batch: int = 2, lr: float = 1e-4, rank: int = 16,
               sample_seconds: float = 3.0) -> str:
    """Train a LoRA adapter and save it to `out`. Returns `out`."""
    captions = load_captions(captions_csv)
    print(f"[train] {len(captions)} captions")

    def meta_fn(info, audio):
        path = os.path.abspath(info.get("path", ""))
        return {"prompt": captions.get(path, "kick drum, single hit, dry")}

    with open(os.path.join(weights_dir, "model_config.json")) as f:
        model_config = json.load(f)
    sample_rate = int(model_config.get("sample_rate", 44100))
    sample_size = max(1, round(sample_seconds * sample_rate / DS_RATIO)) * DS_RATIO
    print(f"[train] sample_rate={sample_rate}; sample_size={sample_size}")
    print(f"[train] loading frozen base from {weights_dir}")
    model = load_diffusion_cond(model_config, os.path.join(weights_dir, "model.safetensors"),
                                device="cpu", model_half=False)

    wrapper = DiffusionCondTrainingWrapper(
        model, lr=lr,
        optimizer_configs={"diffusion": {"optimizer": {"type": "AdamW", "config": {"lr": lr}}}},
        lora_config={"rank": rank, "alpha": rank, "adapter_type": "lora"},
        use_ema=False, timestep_sampler="logit_normal",
        sample_rate=sample_rate, sample_size=sample_size,
        # Required for Stable Audio 3 inpainting checkpoints.
        inpainting_config={"mask_kwargs": {}}, p_one_shot=0.5,
    )

    ds_cfg = LocalDatasetConfig(id="samples", path=os.path.expanduser(src_dir),
                                custom_metadata_fn=meta_fn)
    dataset = SampleDataset([ds_cfg], sample_size=sample_size, sample_rate=sample_rate,
                            random_crop=False, force_channels="stereo", pad=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,
                                         collate_fn=collation_fn, num_workers=0,
                                         drop_last=True)

    trainer = pl.Trainer(
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1, max_steps=steps, precision="32-true",
        enable_checkpointing=False, logger=False, log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(wrapper, loader)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    wrapper.export_lora_safetensors(out)
    print(f"[train] done. LoRA saved -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="Optional JSON training config.")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--sample-seconds", dest="sample_seconds", type=float, default=None)
    args = ap.parse_args()
    cfg = resolve_train_config(args)
    train_lora(cfg["weights"], cfg["captions"], cfg["src"], cfg["out"],
               steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
               rank=cfg["rank"], sample_seconds=cfg["sample_seconds"])


if __name__ == "__main__":
    main()
