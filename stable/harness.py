#!/usr/bin/env python3
"""Repeatable Stable Audio 3 generation harness.

The harness owns prompts, seeds, model/adapter settings, output names, and
structured logs. It is intentionally separate from training so base-vs-LoRA
comparisons can reuse the same generation path.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import torch
import torchaudio

from stable.api import StableAudio, trim_transient
from stable.logging import append_jsonl, utc_now_iso, write_json

DEFAULT_CONFIG = "stable/configs/medium_kick_baseline.json"
DEFAULT_RUN_ROOT = "stable/runs"


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clip_stats(audio: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    peak = float(audio.abs().max().item()) if audio.numel() else 0.0
    rms = float(audio.pow(2).mean().sqrt().item()) if audio.numel() else 0.0
    duration = audio.shape[-1] / sample_rate if audio.dim() > 1 else 0.0
    clipped = bool((audio.abs() >= 0.999).any().item()) if audio.numel() else False
    return {
        "duration_seconds": round(duration, 6),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipped": clipped,
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]) if audio.dim() == 2 else 0,
    }


def resolve_run_dir(run_root: str, config_name: str) -> str:
    return os.path.join(run_root, config_name, timestamp())


def run_harness(
    config_path: str,
    run_root: str,
    limit: int | None = None,
    weights_override: str | None = None,
    lora_override: str | None = None,
    lora_strength_override: float | None = None,
) -> str:
    cfg = load_config(config_path)
    if weights_override:
        cfg["weights"] = weights_override
    if lora_override is not None:
        cfg["lora"] = lora_override
    if lora_strength_override is not None:
        cfg["lora_strength"] = lora_strength_override
    seeds = list(cfg.get("seeds", []))
    if limit is not None:
        seeds = seeds[:limit]
    if not seeds:
        raise ValueError("Harness config must include at least one seed.")

    run_dir = resolve_run_dir(run_root, cfg.get("name", "stable_run"))
    audio_dir = os.path.join(run_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    started = utc_now_iso()
    manifest = {
        "config_path": os.path.abspath(config_path),
        "config": cfg,
        "run_dir": os.path.abspath(run_dir),
        "started_at": started,
        "outputs": [],
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    model = StableAudio(
        cfg.get("weights", "models/stable-audio-3-medium"),
        lora=cfg.get("lora"),
        lora_strength=cfg.get("lora_strength"),
        model_name=cfg.get("model_name", "medium"),
    )

    rows_path = os.path.join(run_dir, "runs.jsonl")
    for index, seed in enumerate(seeds):
        output_path = os.path.join(audio_dir, f"{cfg.get('name', 'sample')}_{index:03d}_seed{seed}.wav")
        row = {
            "index": index,
            "seed": seed,
            "prompt": cfg["prompt"],
            "negative": cfg.get("negative"),
            "weights": cfg.get("weights"),
            "model_name": cfg.get("model_name"),
            "lora": cfg.get("lora"),
            "lora_strength": cfg.get("lora_strength"),
            "seconds": cfg.get("seconds", 1.0),
            "steps": cfg.get("steps", 8),
            "cfg": cfg.get("cfg", 1.5),
            "output_path": os.path.abspath(output_path),
            "started_at": utc_now_iso(),
        }
        t0 = time.perf_counter()
        try:
            audio = model.generate(
                cfg["prompt"],
                negative=cfg.get("negative", "music, melody, loop, reverb"),
                seconds=cfg.get("seconds", 1.0),
                steps=cfg.get("steps", 8),
                cfg=cfg.get("cfg", 1.5),
                seed=seed,
            )
            if cfg.get("trim", False):
                audio = trim_transient(
                    audio,
                    model.sample_rate,
                    floor_ratio=cfg.get("floor_ratio", 0.12),
                )
            audio = audio.clamp(-1, 1)
            torchaudio.save(output_path, audio, model.sample_rate)
            row["status"] = "ok"
            row["audio"] = clip_stats(audio, model.sample_rate)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        row["runtime_seconds"] = round(time.perf_counter() - t0, 3)
        row["finished_at"] = utc_now_iso()
        append_jsonl(rows_path, row)
        manifest["outputs"].append(row)
        write_json(os.path.join(run_dir, "manifest.json"), manifest)
        print(f"[harness] {row['status']} seed={seed} -> {output_path}")

    manifest["finished_at"] = utc_now_iso()
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    print(f"[harness] done -> {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights", default=None, help="Override the config weights path.")
    ap.add_argument("--lora", default=None, help="Override the config LoRA adapter path.")
    ap.add_argument("--lora-strength", type=float, default=None)
    args = ap.parse_args()
    run_harness(
        args.config,
        args.run_root,
        args.limit,
        args.weights,
        args.lora,
        args.lora_strength,
    )


if __name__ == "__main__":
    main()
