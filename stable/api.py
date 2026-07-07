#!/usr/bin/env python3
"""Stable Audio 3 generation API (text -> audio one-shots, + LoRA support).

WHAT IS STABLE AUDIO 3?
    An open-weight text-to-audio latent diffusion model from Stability AI. A
    prompt is encoded by a text encoder (t5gemma) into embedding vectors that
    condition a diffusion transformer (DiT); the DiT denoises a latent, which a
    VAE decodes to 44.1 kHz stereo audio.

    We use the `small-sfx` checkpoint (0.6B) — the sound-effects-tuned variant,
    best for drums/one-shots. It is a `diffusion_cond_inpaint` model: it can
    regenerate a masked region of audio. Plain text-to-audio is the special case
    where the whole clip is masked ("generate everything").

KEY GENERATION SETTINGS (learned empirically for small-sfx)
    - LOW guidance: cfg 1-2. cfg >= 4 over-drives it into dense saturation.
    - It FILLS the requested duration: use short `seconds ~1.0` for a single
      hit; a long duration yields a sustained texture instead of a one-shot.
    - Descriptive/genre prompts ("deep 808 sub kick, single hit, dry") give
      clean decaying hits; a bare "kick drum" gives a sustained drone.

LORA
    A fine-tuned LoRA adapter (see ../stable/train.py) can be applied at load
    time via `lora=`. `lora_strength` blends it (0 = base, 1 = full).

Usage (library):
    from stable.api import StableAudio
    sa = StableAudio("models/stable-audio-3-small-sfx")
    sa.generate_oneshots("deep 808 sub kick, single hit, dry", n=6, out_dir="out")
    # with a trained LoRA:
    sa = StableAudio(".../small-sfx", lora="stable/checkpoints/kicks_lora.safetensors")

Usage (CLI):
    python -m stable.api --prompt "punchy kick, single hit, dry" --n 8 --out out/
    python -m stable.api --prompt "..." --lora stable/checkpoints/kicks_lora.safetensors
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import List, Optional

import torch
import torchaudio
from stable_audio_3 import StableAudioModel

DS_RATIO = 4096  # VAE downsampling; sample_size must be a multiple of this
DEFAULT_WEIGHTS = os.environ.get("SA3_WEIGHTS", "models/stable-audio-3-small-sfx")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def trim_transient(wav: torch.Tensor, sample_rate: int, floor_ratio: float = 0.12,
                   win_ms: float = 10.0, pre_roll_ms: float = 5.0,
                   tail_pad_ms: float = 40.0) -> torch.Tensor:
    """Trim to the hit via a windowed-RMS envelope.

    Robust to the low noise floor of generative output: keep from just before
    the onset through the last window whose RMS exceeds `floor_ratio` of the
    loudest window (plus a short release pad). Higher floor_ratio = tighter.
    wav: (channels, samples), float in [-1, 1].
    """
    mono = wav.abs().mean(dim=0)
    win = max(1, int(sample_rate * win_ms / 1000.0))
    env = mono.unfold(0, win, win).pow(2).mean(dim=1).sqrt()
    if env.numel() == 0 or env.max() <= 0:
        return wav
    active = (env > env.max() * floor_ratio).nonzero()
    if active.numel() == 0:
        return wav
    start = max(0, int(active[0]) * win - int(sample_rate * pre_roll_ms / 1000.0))
    end = min(mono.numel(),
              (int(active[-1]) + 1) * win + int(sample_rate * tail_pad_ms / 1000.0))
    return wav[:, start:end]


class StableAudio:
    """Stable Audio 3 wrapper for prompt->one-shot generation.

    Args:
        weights_dir: local dir with model_config.json + model.safetensors
            (falls back to StableAudioModel.from_pretrained(model_name) if absent).
        device: torch device (auto: cuda -> mps -> cpu).
        lora: optional path to a trained LoRA .safetensors to apply.
        lora_strength: optional LoRA blend (0..~1.5; None = full).
        model_name: hub id used only for the from_pretrained fallback.
    """

    def __init__(self, weights_dir: str = DEFAULT_WEIGHTS, device: Optional[str] = None,
                 lora: Optional[str] = None, lora_strength: Optional[float] = None,
                 model_name: str = "small-sfx"):
        self.device = device or pick_device()
        cfg_path = os.path.join(weights_dir, "model_config.json")
        ckpt = os.path.join(weights_dir, "model.safetensors")
        half = self.device == "cuda"  # half only behaves on CUDA
        if os.path.isfile(cfg_path) and os.path.isfile(ckpt):
            print(f"[sa3] loading local weights from {weights_dir} on {self.device}")
            mod = inspect.getmodule(StableAudioModel)
            with open(cfg_path) as f:
                self.model_config = json.load(f)
            inner = mod.load_diffusion_cond(self.model_config, ckpt,
                                            device=self.device, model_half=half)
            inner.use_lora = False
            inner.lora_names = []
            self.model = StableAudioModel(inner, self.model_config, self.device, half)
        else:
            print(f"[sa3] from_pretrained('{model_name}') on {self.device}")
            self.model = StableAudioModel.from_pretrained(model_name, device=self.device)
            self.model_config = self.model.model_config
        self.sample_rate = self.model_config["sample_rate"]
        if lora:
            print(f"[sa3] applying LoRA: {lora}")
            self.model.load_lora([lora])
            if lora_strength is not None:
                self.model.set_lora_strength(lora_strength)

    def _latent_size(self, seconds: float) -> int:
        return max(1, round(seconds * self.sample_rate / DS_RATIO)) * DS_RATIO

    def generate(self, prompt: str, negative: str = "reverb, room, tail, music, melody, loop",
                 seconds: float = 1.0, steps: int = 8, cfg: float = 1.5,
                 seed: int = 0) -> torch.Tensor:
        """Generate one clip. Returns a (channels, samples) float tensor on CPU."""
        audio = self.model.generate(
            prompt=prompt, negative_prompt=negative, duration=seconds,
            steps=steps, cfg_scale=cfg, batch_size=1,
            sample_size=self._latent_size(seconds), seed=seed,
        )
        audio = audio.detach().to(torch.float32).cpu()
        if audio.dim() == 3:
            audio = audio[0]
        return audio / audio.abs().max().clamp(min=1e-9)  # peak-normalize

    def generate_oneshots(self, prompt: str, n: int = 8, out_dir: str = "samples",
                          seconds: float = 1.0, steps: int = 8, cfg: float = 1.5,
                          seed: int = 0, trim: bool = True,
                          floor_ratio: float = 0.12) -> List[str]:
        """Generate N variations, transient-trim each, save as WAV. Returns paths."""
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i in range(n):
            audio = self.generate(prompt, seconds=seconds, steps=steps, cfg=cfg,
                                   seed=seed + i)
            if trim:
                audio = trim_transient(audio, self.sample_rate, floor_ratio=floor_ratio)
            audio = audio.clamp(-1, 1)
            path = os.path.join(out_dir, f"oneshot_{i:03d}.wav")
            torchaudio.save(path, audio, self.sample_rate)
            paths.append(path)
            print(f"[sa3] wrote {path}  ({audio.shape[-1] / self.sample_rate:.2f}s)")
        return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.12)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-strength", type=float, default=None)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()

    sa = StableAudio(args.weights, lora=args.lora, lora_strength=args.lora_strength)
    sa.generate_oneshots(args.prompt, n=args.n, out_dir=args.out, seconds=args.seconds,
                         steps=args.steps, cfg=args.cfg, seed=args.seed,
                         trim=not args.no_trim, floor_ratio=args.floor)
    print(f"[sa3] done -> {args.out}/")


if __name__ == "__main__":
    main()
