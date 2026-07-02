# Hybrid Text→Audio MVP (MusicGen → AudioSR)

A local, **no-training, no-gated-login** text-to-audio pipeline that generates
stereo music from a text prompt and then restores/upscales it to a 48 kHz
master — by chaining two published, fully-open pretrained models:

1. **Stage 1 — Generation:** [`facebook/musicgen-stereo-small`](https://huggingface.co/facebook/musicgen-stereo-small)
   (Meta MusicGen). Text prompt → native **stereo 32 kHz** audio. Open/ungated.
2. **Stage 2 — Restoration/Upscale:** [`HaoheLiu/AudioSR`](https://github.com/haoheliu/versatile_audio_super_resolution).
   Latent-diffusion super-resolution → **stereo 48 kHz** master, with
   bandwidth extension (synthesized highs) and artifact cleanup.

The stages hand off via a WAV file on disk (`raw_stage1.wav` → `final_sota_sample.wav`).

> This is a separate MVP from the `audio_restoration_benchmark.py` project
> documented in [README.md](README.md). It shares the AudioSR dependency.

---

## ⚠️ Honest status — what is and isn't true here

- ✅ **Verified end-to-end on Apple Silicon (MPS).** A real prompt
  (*"warm lofi hip hop, mellow rhodes, vinyl crackle"*) produced genuine
  generated stereo audio through both stages. This is not a mock.
- ⚠️ **AudioSR is diffusion (DDIM), not "flow matching."** Earlier framing in
  this effort called Stage 2 a "flow-matching super-resolution" — that's
  marketing. AudioSR is a latent-diffusion model. There is no public model
  called "LavaSR."
- ⚠️ **AudioSR quantizes output duration.** It pads to its internal segment
  length, so a 4 s request comes out ~5.12 s (the tail is padding). Trim
  downstream if you need exact length.
- ⚠️ **Stereo is processed per-channel** (AudioSR is mono-oriented). Running L
  and R through independent diffusion can slightly loosen the stereo image.
  Fine for an MVP; mid/side processing would tighten it.
- ⚠️ **MusicGen ignores the negative prompt** and has no "inference steps"
  (it's autoregressive; length = tokens at ~50/sec). The Gradio negative-prompt
  field is kept for UI symmetry but is a no-op.
- ⚠️ **Quality is a step below a large diffusion model.** `-small` is fast and
  light; swap to `-medium`/`-large` for better fidelity (see below).
- ℹ️ **License:** MusicGen weights are CC-BY-NC (non-commercial).

---

## Why two virtual environments?

The two stages have **mutually incompatible dependencies** and cannot share one
process:

| | Stage 1 (`.venv-stage1`) | Stage 2 (`.venv-stage2`) |
|---|---|---|
| Python | 3.13 | **3.11** (audiosr won't build on 3.12/3.13) |
| Key pins | modern `transformers`, `torch` | `torch==2.2.2`, `numpy==1.23.5`, `matplotlib<3.8` |
| Runs | in-process | as a **subprocess** (`stage2_audiosr_worker.py`) |

`audiosr` hard-pins `numpy<=1.23.5` and breaks on `torch>=2.9`
(new `torchaudio.load` requires `torchcodec`), which is irreconcilable with the
modern MusicGen stack. So Stage 2 lives in its own venv and is invoked over a
file handoff. If `.venv-stage2` is missing, Stage 2 **auto-falls back** to a
plain high-quality resample to 48 kHz.

---

## Setup

Requires macOS/Linux with Python 3.13 **and** Python 3.11 available
(`brew install python@3.11` on macOS). CUDA or Apple MPS auto-detected; CPU
works but AudioSR is slow on CPU.

```bash
# --- Stage 1 venv: MusicGen (no gated login needed) ---
python3.13 -m venv .venv-stage1
./.venv-stage1/bin/pip install --upgrade pip
./.venv-stage1/bin/pip install torch torchaudio transformers accelerate sentencepiece soundfile numpy gradio

# --- Stage 2 venv: AudioSR (pinned, Python 3.11) ---
python3.11 -m venv .venv-stage2
./.venv-stage2/bin/pip install --upgrade pip setuptools wheel
./.venv-stage2/bin/pip install audiosr
./.venv-stage2/bin/pip install "torch==2.2.2" "torchaudio==2.2.2" "torchvision==0.17.2" "numpy==1.23.5" "matplotlib<3.8"
```

Both venvs and the generated `*.wav` files are git-ignored.

---

## Usage

Run **from the Stage-1 venv**; it shells out to the Stage-2 venv automatically.

```bash
# Web UI (Gradio)
./.venv-stage1/bin/python hybrid_audio_engine.py

# CLI
./.venv-stage1/bin/python hybrid_audio_engine.py --mode cli \
    --prompt "warm lofi hip hop, mellow rhodes, vinyl crackle" \
    --duration 4 --sr-steps 8
```

Output: `final_sota_sample.wav` (stereo, 48 kHz). Intermediate: `raw_stage1.wav`.

### Knobs
- `--duration` seconds of music to generate (MusicGen).
- `--gen-guidance` MusicGen classifier-free guidance (default 3.0).
- `--sr-steps` AudioSR DDIM steps (default 8; raise toward 20–50 for a cleaner,
  less grainy master at the cost of speed).
- **Bigger model:** edit `STAGE1_MODEL` in `hybrid_audio_engine.py`
  (`facebook/musicgen-stereo-medium` / `-large`, or `musicgen-small` for mono).

---

## Files

| File | Role |
|---|---|
| `hybrid_audio_engine.py` | Stage-1 (MusicGen) + orchestration + Gradio/CLI |
| `stage2_audiosr_worker.py` | Stage-2 AudioSR worker, runs in the 3.11 venv |
| `sota_audio_gen.py` | **Reference only** — an earlier WavTokenizer + CFM-DiT template. The codec half works; the CFM generative core is an untrained scaffold (no public checkpoint generates in WavTokenizer's token space). Kept as a learning artifact, **not** part of the working pipeline. |

---

## Hardware

Auto-detects **CUDA → Apple MPS → CPU**. Stage 1 uses fp16 on CUDA, fp32 on
MPS/CPU. AudioSR runs on CUDA if present, otherwise CPU (its MPS coverage is
incomplete), so on Apple Silicon Stage 2 is the slow part (~1 min per few
seconds of audio at 8 steps).
