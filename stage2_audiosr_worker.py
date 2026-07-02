#!/usr/bin/env python3
# =============================================================================
#  Stage-2 worker — runs INSIDE the Python 3.11 `.venv-stage2` (audiosr).
#  Invoked as a subprocess by hybrid_audio_engine.py because audiosr's pins
#  (numpy 1.23.5 / transformers 4.30.2) are incompatible with the Stage-1
#  Stable Audio Open stack. Handoff is via WAV files on disk.
#
#  Reads a stereo (or mono) WAV, runs AudioSR per-channel (AudioSR is mono-
#  oriented), and writes a 48kHz result.
#
#  Usage (normally you don't call this directly):
#    .venv-stage2/bin/python stage2_audiosr_worker.py \
#        --input raw_stage1.wav --output final_sota_sample.wav --steps 8
# =============================================================================
import argparse
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from audiosr import build_model, super_resolution

FINAL_SR = 48_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--model", default="basic")   # 'basic' or 'speech'
    args = ap.parse_args()

    # AudioSR's MPS coverage is incomplete; use CUDA if present, else CPU.
    sr_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage2] device={sr_device} model={args.model} steps={args.steps}")
    model = build_model(model_name=args.model, device=sr_device)

    data, sr = sf.read(args.input, dtype="float32", always_2d=True)   # [N, C]
    chans = []
    for c in range(data.shape[1]):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            mono_path = tf.name
        sf.write(mono_path, data[:, c], sr)
        wav = super_resolution(
            model, mono_path,
            seed=42, ddim_steps=int(args.steps), guidance_scale=float(args.guidance),
        )
        os.remove(mono_path)
        chans.append(np.asarray(wav, dtype=np.float32).squeeze())

    n = min(len(x) for x in chans)
    out = np.stack([x[:n] for x in chans], axis=1)   # [N, C] @ 48kHz
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 1.0:
        out = out / peak
    sf.write(args.output, out, FINAL_SR)
    print(f"[stage2] wrote {args.output} shape={out.shape} @ {FINAL_SR}Hz")


if __name__ == "__main__":
    main()
