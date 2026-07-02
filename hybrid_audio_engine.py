#!/usr/bin/env python3
# =============================================================================
#  HybridAudioEngine
#  Local MVP: text -> Stereo 32kHz music (MusicGen) -> 48kHz master
#  via a diffusion super-resolution / restoration stage (AudioSR).
#
#  NO TRAINING and NO GATED LOGIN REQUIRED — both stages use fully open,
#  published pretrained weights that download without accepting any license.
#
#  ---------------------------------------------------------------------------
#  HONEST NOTES (read once, save yourself an hour):
#   * Stage 1 "facebook/musicgen-stereo-small" is an OPEN, ungated model
#     (Meta, CC-BY-NC). It downloads with no HF login. It is autoregressive
#     (not diffusion): "steps" don't apply; length is controlled by tokens
#     (~50 tokens/sec). It does NOT support a negative prompt — that field is
#     accepted for UI compatibility but ignored. Swap the model id to
#     "facebook/musicgen-small" for mono, or "-medium"/"-large" for size.
#   * Stage 2 uses the REAL "HaoheLiu/AudioSR". It is a *latent diffusion*
#     (DDIM) super-resolution model — not literally "flow matching", and there
#     is no public model called "LavaSR". What AudioSR actually does well:
#     bandwidth extension to a full 48kHz band (restores/synthesizes highs)
#     and general artifact cleanup. It is mono-oriented, so for stereo we run
#     it per-channel (L/R independently) — good enough for an MVP, with a small
#     imaging caveat. Fewer DDIM steps = faster but rougher; default is modest.
#   * If AudioSR can't be imported, the engine falls back to a high-quality
#     polyphase resample to 48kHz so the pipeline still yields a master file.
# =============================================================================

# -----------------------------------------------------------------------------
#  1. ENVIRONMENT SETUP  (run once)
# -----------------------------------------------------------------------------
#   # CUDA users: install the torch build matching your CUDA from pytorch.org.
#   # Apple Silicon / CPU: default wheels below are fine.
#   #
#   # Stage 1 (this venv): MusicGen via transformers — NO gated login needed.
#   pip install --upgrade torch torchaudio
#   pip install --upgrade transformers accelerate sentencepiece
#   pip install soundfile numpy gradio
#   #
#   # Stage 2 lives in a SEPARATE Python 3.11 venv (audiosr's pins are
#   # incompatible with the modern stack). See stage2_audiosr_worker.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import subprocess

import numpy as np
import torch
import torchaudio
import soundfile as sf

# Two-environment layout (see stage2_audiosr_worker.py for why):
#   Stage 1 (this process) : MusicGen / transformers        -> .venv-stage1
#   Stage 2 (subprocess)   : audiosr (py3.11, old pins)     -> .venv-stage2
_HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_PYTHON = os.path.join(_HERE, ".venv-stage2", "bin", "python")
STAGE2_WORKER = os.path.join(_HERE, "stage2_audiosr_worker.py")

STAGE1_MODEL = "facebook/musicgen-stereo-small"    # open/ungated; mono: musicgen-small; bigger: -medium/-large
MUSICGEN_FPS = 50                                  # MusicGen frame rate (tokens/sec)


# =============================================================================
#  2. HARDWARE DETECTION
# =============================================================================
def pick_device() -> str:
    if torch.cuda.is_available():
        print(f"[hw] CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("[hw] Apple Silicon (MPS)")
        return "mps"
    print("[hw] CPU")
    return "cpu"


STAGE1_SR = 32_000     # MusicGen native sample rate (set from model at load)
FINAL_SR = 48_000      # target master sample rate


# =============================================================================
#  3. PIPELINE CORE
# =============================================================================
class HybridAudioEngine:
    def __init__(self, device: str | None = None):
        self.device = device or pick_device()
        # float16 pays off on CUDA; MPS/CPU are more stable in float32.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.gen_model = None      # MusicgenForConditionalGeneration (Stage 1)
        self.gen_proc = None       # AutoProcessor
        self.stage1_sr = STAGE1_SR
        # Stage 2 lives in a SEPARATE venv/process; we detect its worker here.
        self.stage2_available = os.path.isfile(STAGE2_PYTHON) and os.path.isfile(STAGE2_WORKER)

    # ---- 3a. Load Stage-1; report Stage-2 availability ----------------------
    def load_models(self):
        # --- Stage 1: MusicGen (open, ungated, autoregressive) ---
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        print(f"[load] MusicGen (open, ungated): {STAGE1_MODEL}")
        self.gen_proc = AutoProcessor.from_pretrained(STAGE1_MODEL)
        self.gen_model = MusicgenForConditionalGeneration.from_pretrained(
            STAGE1_MODEL, torch_dtype=self.dtype
        ).to(self.device)
        self.stage1_sr = self.gen_model.config.audio_encoder.sampling_rate  # 32000

        # --- Stage 2: AudioSR runs out-of-process (incompatible deps) ---
        if self.stage2_available:
            print(f"[load] Stage-2 AudioSR worker detected: {STAGE2_PYTHON}")
        else:
            print("[load] Stage-2 AudioSR venv not found "
                  f"(expected {STAGE2_PYTHON}). Stage 2 will fall back to "
                  "high-quality resampling to 48kHz.")
        return self

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _to_ch_last(audio: np.ndarray) -> np.ndarray:
        """Coerce to [samples, channels] float32 for soundfile."""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        elif a.shape[0] < a.shape[1]:   # looks like [channels, samples]
            a = a.T
        return a

    def _resample_fallback(self, in_path: str) -> str:
        """Clean polyphase resample to 48kHz when AudioSR is unavailable."""
        data, sr = sf.read(in_path, dtype="float32", always_2d=True)   # [N, C]
        t = torchaudio.functional.resample(torch.from_numpy(data.T), sr, FINAL_SR)
        out = t.T.numpy()
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 1.0:
            out = out / peak
        return out, FINAL_SR

    def _super_resolve(self, in_path: str, out_path: str, steps: int, guidance: float) -> str:
        """
        Stage 2. If the audiosr venv exists, run the worker as a subprocess
        (real AudioSR, per-channel, 48kHz). Otherwise resample-fallback.
        Returns the path to the written 48kHz file.
        """
        if self.stage2_available:
            cmd = [
                STAGE2_PYTHON, STAGE2_WORKER,
                "--input", in_path, "--output", out_path,
                "--steps", str(int(steps)), "--guidance", str(float(guidance)),
            ]
            print(f"[sr] launching Stage-2 worker: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"Stage-2 AudioSR worker failed (exit {proc.returncode})")
            return out_path

        # --- fallback: resample and write directly ---
        out, out_sr = self._resample_fallback(in_path)
        sf.write(out_path, out, out_sr)
        return out_path

    # ---- 3b. Main hybrid generation -----------------------------------------
    @torch.no_grad()
    def generate_hybrid(
        self,
        prompt: str,
        negative_prompt: str = "",       # accepted for UI compat; MusicGen ignores it
        duration: float = 8.0,
        gen_guidance: float = 3.0,       # MusicGen classifier-free guidance scale
        sr_steps: int = 8,
        sr_guidance: float = 3.5,
        stage1_path: str = "raw_stage1.wav",
        final_path: str = "final_sota_sample.wav",
    ) -> str:
        assert self.gen_model is not None, "call load_models() first"

        # --- Step 1: MusicGen -> raw stereo 32kHz ---
        print(f"[gen] MusicGen: '{prompt}' ({duration:.1f}s)")
        max_new_tokens = max(1, int(round(duration * MUSICGEN_FPS)))
        inputs = self.gen_proc(text=[prompt], padding=True, return_tensors="pt").to(self.device)
        audio_values = self.gen_model.generate(
            **inputs, do_sample=True, guidance_scale=float(gen_guidance),
            max_new_tokens=max_new_tokens,
        )
        audio = audio_values[0].to(torch.float32).cpu().numpy()    # [channels, samples]
        audio = self._to_ch_last(audio)                            # [samples, channels]
        sf.write(stage1_path, audio, self.stage1_sr)
        print(f"[gen] wrote {stage1_path}  shape={audio.shape} @ {self.stage1_sr}Hz")

        # --- Step 2: super-resolution / restoration -> 48kHz master ---
        mode = "AudioSR (subprocess)" if self.stage2_available else "resample-fallback"
        print(f"[sr] {mode}: -> {FINAL_SR}Hz stereo (steps={sr_steps})")
        self._super_resolve(stage1_path, final_path, sr_steps, sr_guidance)
        print(f"[done] wrote {final_path}")
        return final_path


# =============================================================================
#  4. GRADIO WEB UI
# =============================================================================
def launch_gradio(engine: HybridAudioEngine):
    import gradio as gr

    def _run(prompt, negative, duration):
        return engine.generate_hybrid(
            prompt=prompt, negative_prompt=negative, duration=float(duration),
            final_path="final_sota_sample.wav",
        )

    with gr.Blocks(title="HybridAudioEngine — MusicGen + AudioSR") as demo:
        gr.Markdown("## Hybrid Text→Audio\nMusicGen (32kHz stereo) → AudioSR restoration (48kHz master)")
        prompt = gr.Textbox(label="Prompt", value="warm cinematic ambient pad, deep sub bass, analog warmth")
        negative = gr.Textbox(label="Negative prompt (ignored by MusicGen)", value="")
        duration = gr.Slider(1, 10, value=8, step=1, label="Duration (seconds)")
        btn = gr.Button("Generate SOTA Sample", variant="primary")
        out = gr.Audio(label="final_sota_sample.wav", type="filepath")
        btn.click(_run, [prompt, negative, duration], out)

    demo.launch()


# =============================================================================
#  5. ENTRY POINT
# =============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description="Hybrid MusicGen + AudioSR MVP")
    p.add_argument("--mode", choices=["gradio", "cli"], default="gradio")
    p.add_argument("--prompt", default="warm cinematic ambient pad, deep sub bass, analog warmth")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--gen-guidance", type=float, default=3.0)
    p.add_argument("--sr-steps", type=int, default=8)
    args = p.parse_args()

    engine = HybridAudioEngine().load_models()

    if args.mode == "gradio":
        launch_gradio(engine)
    else:
        engine.generate_hybrid(
            prompt=args.prompt, duration=args.duration,
            gen_guidance=args.gen_guidance, sr_steps=args.sr_steps,
        )


if __name__ == "__main__":
    main()
