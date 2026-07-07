#!/usr/bin/env python3
"""CLAP audio-tagging API.

WHAT IS CLAP?
    CLAP (Contrastive Language-Audio Pretraining) is a joint audio<->text
    embedding model. Its audio tower (an HTSAT-style encoder) and text tower
    (RoBERTa) are trained so that a clip and a matching description land at
    *nearby* points in one shared vector space. That lets you do zero-shot
    tagging: embed an audio clip, embed a list of candidate descriptor phrases,
    and rank the descriptors by cosine similarity to the audio.

    Note: CLAP's text encoder is RoBERTa. This is a *different* model from
    Stable Audio 3's text encoder (t5gemma) — the two are unrelated. Both are
    text *encoders* (they output vectors), not text generators.

HOW WE USE IT
    Every file in the target library is already a kick, so we don't classify
    "kick vs not". Instead we attach acoustic *descriptors* (808, punchy,
    distorted, sub bass, ...) to each file and combine them with a cleaned
    filename to produce a training caption for the Stable Audio 3 LoRA
    (see ../stable/train.py). Output is a `captions.csv` (filepath,prompt).

THREE COMPATIBILITY PATCHES (why they exist)
    The pinned `laion_clap` predates our torch/transformers versions, so
    `ClapTagger` applies three fixes on load:
      1. torch.load -> weights_only=False   (checkpoint has numpy globals;
         torch>=2.6 defaults to weights_only=True and would reject it)
      2. load_state_dict(strict=False)      (transformers>=5 dropped RoBERTa's
         `position_ids` buffer that the checkpoint still carries)
      3. audio passed as a torch tensor when use_tensor=True

DECODING
    Ableton factory .aif use a proprietary `able` codec that libsndfile/ffmpeg
    can't read; convert to WAV first. WAV/AIFF-PCM decode via ffmpeg here.

Usage (library):
    from clap.api import ClapTagger
    tagger = ClapTagger()
    print(tagger.tag("kick.wav"))       # ['punchy', 'tight', 'sub bass', '808']
    print(tagger.caption("kick.wav"))   # 'kick drum, ..., single hit, dry'
    tagger.tag_directory("~/Music/Kicks", "clap/captions.csv")

Usage (CLI):
    python -m clap.api --src ~/Music/Kicks --out clap/captions.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
from typing import List, Optional

import numpy as np
import torch

# --- patch 1: laion_clap's checkpoint predates torch 2.6's weights_only default.
_orig_torch_load = torch.load
def _load_full(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _load_full

import laion_clap  # noqa: E402  (import after the torch.load patch)

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
CLAP_SR = 48000

# Default acoustic-descriptor vocabulary for kick one-shots.
DEFAULT_TAGS = [
    "808", "sub bass", "deep", "boomy", "punchy", "tight", "snappy", "clicky",
    "distorted", "saturated", "gritty", "clean", "analog", "acoustic",
    "synthetic", "lo-fi", "hard", "soft", "dark", "bright", "long tail", "short",
]


def decode_mono_48k(path: str) -> np.ndarray:
    """Decode any ffmpeg-readable audio to float32 mono @ 48 kHz (CLAP's rate)."""
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(CLAP_SR),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def clean_name(path: str) -> str:
    """Turn a filename into a readable caption fragment (drop indices/pack cruft)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\b\d{1,4}\b", "", stem)
    stem = re.sub(r"\bSR\b|\bMPL\b|\bAmp\b", "", stem, flags=re.I)
    return re.sub(r"\s+", " ", stem).strip(" -_").lower()


class ClapTagger:
    """Zero-shot acoustic tagger built on laion_clap.

    Args:
        tags: descriptor vocabulary to rank against (defaults to DEFAULT_TAGS).
        topk: how many descriptors to keep per file.
        device: torch device (auto: cuda -> mps -> cpu).
    """

    def __init__(self, tags: Optional[List[str]] = None, topk: int = 4,
                 device: Optional[str] = None):
        self.tags = tags or DEFAULT_TAGS
        self.topk = topk
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        # patch 2: tolerate the stale RoBERTa position_ids key.
        _orig_lsd = self.model.model.load_state_dict
        self.model.model.load_state_dict = lambda sd, strict=True: _orig_lsd(sd, strict=False)
        self.model.load_ckpt()  # downloads the 630k checkpoint on first run
        # Precompute normalized text embeddings for the vocabulary.
        prompts = [f"a {t} kick drum" for t in self.tags]
        te = self.model.get_text_embedding(prompts, use_tensor=True)
        self._tag_embed = torch.nn.functional.normalize(te, dim=-1)

    def embed_audio(self, path: str) -> torch.Tensor:
        """Return the L2-normalized CLAP embedding for one audio file."""
        audio = decode_mono_48k(path)
        # patch 3: use_tensor=True requires a torch tensor input.
        emb = self.model.get_audio_embedding_from_data(
            x=torch.from_numpy(audio[None, :]), use_tensor=True)
        return torch.nn.functional.normalize(emb, dim=-1)

    def tag(self, path: str) -> List[str]:
        """Return the top-k descriptor tags for one file, most similar first."""
        sims = (self.embed_audio(path) @ self._tag_embed.T).squeeze(0)
        return [self.tags[j] for j in sims.topk(self.topk).indices.tolist()]

    def caption(self, path: str) -> str:
        """Build a training caption: cleaned filename + CLAP descriptors."""
        base = clean_name(path)
        desc = ", ".join(dict.fromkeys(self.tag(path)))
        cap = f"kick drum, {base + ', ' if base else ''}{desc}, single hit, dry"
        return re.sub(r"\s+", " ", cap).strip()

    def tag_directory(self, src: str, out_csv: str, limit: int = 0) -> list:
        """Tag every audio file under `src`, writing a filepath,prompt CSV.

        Returns the list of (filepath, caption) rows.
        """
        src = os.path.expanduser(src)
        files = sorted(
            glob.glob(os.path.join(src, "**", "*.wav"), recursive=True) +
            glob.glob(os.path.join(src, "**", "*.aif"), recursive=True) +
            glob.glob(os.path.join(src, "**", "*.aiff"), recursive=True))
        if limit:
            files = files[:limit]
        print(f"[clap] tagging {len(files)} files on {self.device}")
        rows = []
        for i, f in enumerate(files):
            try:
                rows.append((f, self.caption(f)))
            except Exception as e:  # skip undecodable / corrupt files
                print(f"[clap] skip {os.path.basename(f)}: {e}")
                continue
            if i < 8 or i % 50 == 0:
                print(f"[clap] {os.path.basename(f)[:34]:34s} -> {rows[-1][1]}")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filepath", "prompt"])
            w.writerows(rows)
        print(f"[clap] wrote {len(rows)} captions -> {out_csv}")
        return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="CLAP-tag an audio folder into captions.csv")
    ap.add_argument("--src", required=True, help="root dir of audio samples")
    ap.add_argument("--out", default="clap/captions.csv")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="tag only N files (0=all)")
    args = ap.parse_args()
    ClapTagger(topk=args.topk).tag_directory(args.src, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
