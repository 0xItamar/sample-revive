#!/usr/bin/env python3
# =============================================================================
#  SOTAAudioGenPipeline
#  A local MVP of a 2025/2026-style Text-to-Audio pipeline:
#     WavTokenizer (single-quantizer neural codec)  +
#     Conditional Flow-Matching (CFM) Diffusion Transformer (DiT) core.
#
#  ---------------------------------------------------------------------------
#  READ THIS FIRST — what is real vs. what is a template
#  ---------------------------------------------------------------------------
#  REAL / WORKING out of the box:
#    * WavTokenizer encode + decode (extreme compression, ~40-75 tok/s).
#      Load a reference .wav -> tokens -> reconstruct .wav. This is trained.
#    * Text encoder (CLAP by default) producing a real conditioning vector.
#
#  TEMPLATE / UNTRAINED (clearly flagged below with `!!! UNTRAINED !!!`):
#    * The CFM-DiT generative core. The architecture, adaLN conditioning,
#      in-context reference conditioning, and the flow-matching Euler ODE
#      solver are all implemented correctly and production-grade — BUT the
#      weights are randomly initialized. There is NO public checkpoint of a
#      CFM model that lives in WavTokenizer's latent space. Until you train
#      (or fine-tune) this core, `generate_from_prompt_and_reference` will
#      run end-to-end and emit a WAV, but the *generated* portion is noise
#      shaped like audio, not meaningful new content.
#
#  Want real output TODAY? Two production paths, documented at the bottom:
#    (A) Use this file purely as a WavTokenizer codec demo (--mode reconstruct).
#    (B) Swap the DiT for F5-TTS's trained CFM-DiT (mel space). Pointers below.
# =============================================================================

# -----------------------------------------------------------------------------
#  1. ENVIRONMENT & INSTALL  (run these once in your shell)
# -----------------------------------------------------------------------------
#
#   # --- clone the WavTokenizer repo (provides encoder/ and decoder/ modules) ---
#   git clone https://github.com/jishengpeng/WavTokenizer.git
#   cd WavTokenizer
#
#   # --- core ML stack ---
#   #  CUDA users: install the torch build matching your CUDA (see pytorch.org).
#   #  Apple Silicon / CPU: the default wheels below are fine.
#   pip install torch torchaudio
#   pip install transformers einops soundfile numpy pyyaml
#   pip install gradio                      # optional, only for the GUI
#
#   # WavTokenizer's own requirements (vocos, etc.):
#   pip install -r requirements.txt         # from inside the cloned repo
#   # If requirements.txt is missing a couple things, these cover the codec:
#   pip install vocos encodec
#
#   # --- run this script from *inside* the cloned WavTokenizer dir ---
#   #  (so that `from encoder...` / `from decoder...` imports resolve),
#   #  OR pass --wavtokenizer-repo /path/to/WavTokenizer .
#
# -----------------------------------------------------------------------------
#  CHECKPOINTS TO DOWNLOAD (WavTokenizer — required for real codec output)
# -----------------------------------------------------------------------------
#   Recommended (the "75-token/s large" model you named):
#     * Config : WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_...yaml
#     * Weights: from HF hub  ->  novateur/WavTokenizer-large-speech-75token
#                (file: wavtokenizer_large_speech_320_24k.ckpt  or similar)
#
#   Download example:
#     pip install huggingface_hub
#     huggingface-cli download novateur/WavTokenizer-large-speech-75token \
#         --local-dir ./wavtokenizer_ckpts
#
#   Then point --wt-config and --wt-ckpt at the yaml + .ckpt you downloaded.
#   (The config yaml ships inside the cloned GitHub repo under configs/.)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# =============================================================================
#  2. HARDWARE DETECTION
# =============================================================================
def pick_device() -> torch.device:
    """CUDA > Apple MPS > CPU, with a friendly banner."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
        name = "Apple Silicon (MPS)"
    else:
        dev = torch.device("cpu")
        name = "CPU"
    print(f"[hw] Using device: {dev}  ({name})")
    return dev


# WavTokenizer operates at 24 kHz, mono.
WT_SAMPLE_RATE = 24_000


# =============================================================================
#  3. THE CFM-DiT GENERATIVE CORE   !!! ARCHITECTURE REAL, WEIGHTS UNTRAINED !!!
# =============================================================================
#  This is a compact-but-faithful Diffusion Transformer with:
#     * sinusoidal flow-time embedding
#     * adaLN-Zero conditioning (DiT-style) from a global condition vector
#       (= text embedding [+ pooled reference])
#     * in-context reference conditioning: the clean reference latents are
#       concatenated as extra channels so the model "hears" the timbre while
#       predicting the velocity field for the generated region.
#     * a proper Conditional Flow Matching objective + Euler ODE sampler
#       (4-8 steps), i.e. rectified-flow style straight-line transport.
# -----------------------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of a continuous flow-time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (Peebles & Xie, 2023)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # adaLN-Zero: produce 6 modulation params from the global condition.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa_sh, sa_sc, sa_g, mlp_sh, mlp_sc, mlp_g = self.ada(c).chunk(6, dim=-1)
        h = self._mod(self.norm1(x), sa_sh, sa_sc)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + sa_g.unsqueeze(1) * h
        h = self._mod(self.norm2(x), mlp_sh, mlp_sc)
        x = x + mlp_g.unsqueeze(1) * self.mlp(h)
        return x


class CFMDiT(nn.Module):
    """
    Conditional Flow-Matching Diffusion Transformer over WavTokenizer *latents*.

    Works in the continuous feature space that WavTokenizer's decoder consumes
    (shape [B, C, T]), which is the natural home for flow matching — we predict
    a velocity field and integrate it, then hand the continuous latents straight
    to the codec decoder (no discrete resampling needed).
    """

    def __init__(self, latent_dim: int, cond_dim: int, model_dim: int = 512,
                 depth: int = 8, heads: int = 8, max_len: int = 4096):
        super().__init__()
        self.latent_dim = latent_dim
        # x_t (latent_dim) + clean reference latent (latent_dim, in-context) -> model_dim
        self.in_proj = nn.Linear(latent_dim * 2, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.t_embed = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(),
                                     nn.Linear(model_dim, model_dim))
        self.cond_proj = nn.Linear(cond_dim, model_dim)

        self.blocks = nn.ModuleList([DiTBlock(model_dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(model_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.model_dim = model_dim

    def forward(self, x_t, t, cond_vec, ref_latent):
        """
        x_t        : [B, T, latent_dim]  current point on the flow path
        t          : [B]                 flow-time in [0, 1]
        cond_vec   : [B, cond_dim]       global condition (text [+ ref pooled])
        ref_latent : [B, T, latent_dim]  clean reference latents (in-context)
        returns velocity field v_theta : [B, T, latent_dim]
        """
        h = self.in_proj(torch.cat([x_t, ref_latent], dim=-1))
        h = h + self.pos[:, : h.size(1)]
        c = self.t_embed(timestep_embedding(t, self.model_dim)) + self.cond_proj(cond_vec)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_proj(self.norm_out(h))

    @torch.no_grad()
    def sample(self, cond_vec, ref_latent, gen_len, steps=6):
        """
        Solve the flow-matching ODE from noise (t=0) to data (t=1) with a
        straight Euler integrator. `steps` in [4, 8] is the SOTA fast regime.

        In-context infilling: the reference region stays clean; we only
        integrate the newly generated region (length `gen_len`).
        """
        assert 4 <= steps <= 10, "Fast CFM sampling is designed for 4-10 steps."
        B = cond_vec.size(0)
        dev = cond_vec.device
        ref_len = ref_latent.size(1)
        total = ref_len + gen_len

        # Pad reference latents out to full length (gen region = zeros placeholder).
        ref_full = F.pad(ref_latent, (0, 0, 0, gen_len))  # [B, total, D]

        # Start the generated region from Gaussian noise; keep ref region as ref.
        x = torch.randn(B, total, self.latent_dim, device=dev)
        x[:, :ref_len] = ref_latent

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=dev)
            v = self.forward(x, t, cond_vec, ref_full)
            x = x + dt * v
            x[:, :ref_len] = ref_latent  # re-clamp the in-context reference
        return x[:, ref_len:]  # only the freshly generated latents [B, gen_len, D]


# =============================================================================
#  4. THE PIPELINE
# =============================================================================
@dataclass
class PipelineConfig:
    wt_config: str            # path to WavTokenizer .yaml
    wt_ckpt: str              # path to WavTokenizer .ckpt
    text_model: str = "laion/clap-htsat-unfused"   # CLAP; or a T5 id
    cfm_ckpt: str | None = None    # your trained CFMDiT weights (.pt); None => untrained
    tokens_per_second: int = 75    # WavTokenizer large "75-token" frame rate


class SOTAAudioGenPipeline:
    def __init__(self, cfg: PipelineConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or pick_device()
        self.wavtokenizer = None
        self.text_model = None
        self.text_proc = None
        self.cfm = None
        self._cond_dim = None
        self._latent_dim = None

    # ---- 4a. WavTokenizer encoder + decoder ---------------------------------
    def load_codec(self):
        """Load the pretrained WavTokenizer (encoder + decoder in one module)."""
        try:
            from decoder.pretrained import WavTokenizer  # from the cloned repo
        except ImportError as e:
            raise ImportError(
                "Could not import WavTokenizer. Run this script from inside the "
                "cloned jishengpeng/WavTokenizer repo, or add it with "
                "--wavtokenizer-repo /path/to/WavTokenizer."
            ) from e

        print(f"[codec] Loading WavTokenizer\n         config: {self.cfg.wt_config}\n"
              f"         ckpt  : {self.cfg.wt_ckpt}")
        self.wavtokenizer = WavTokenizer.from_pretrained0802(
            self.cfg.wt_config, self.cfg.wt_ckpt
        ).to(self.device).eval()
        return self

    # ---- 4b. Text encoder (CLAP by default, T5 optional) --------------------
    def load_text_encoder(self):
        mid = self.cfg.text_model
        if "clap" in mid.lower():
            from transformers import ClapModel, ClapProcessor
            print(f"[text] Loading CLAP text encoder: {mid}")
            self.text_model = ClapModel.from_pretrained(mid).to(self.device).eval()
            self.text_proc = ClapProcessor.from_pretrained(mid)
            self._cond_dim = self.text_model.config.projection_dim  # 512
            self._text_kind = "clap"
        else:
            from transformers import T5EncoderModel, AutoTokenizer
            print(f"[text] Loading T5 text encoder: {mid}")
            self.text_model = T5EncoderModel.from_pretrained(mid).to(self.device).eval()
            self.text_proc = AutoTokenizer.from_pretrained(mid)
            self._cond_dim = self.text_model.config.d_model
            self._text_kind = "t5"
        return self

    @torch.no_grad()
    def embed_text(self, prompt: str) -> torch.Tensor:
        """Return a single global condition vector [1, cond_dim]."""
        if self._text_kind == "clap":
            inp = self.text_proc(text=[prompt], return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            return self.text_model.get_text_features(**inp)          # [1, 512]
        else:
            inp = self.text_proc([prompt], return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            seq = self.text_model(**inp).last_hidden_state           # [1, L, d]
            return seq.mean(dim=1)                                   # mean-pool -> [1, d]

    # ---- 4c. CFM-DiT core ---------------------------------------------------
    def load_cfm(self, latent_dim: int):
        """
        Build the CFM-DiT. If cfg.cfm_ckpt is given, load trained weights;
        otherwise leave it randomly initialized (template mode).
        """
        self._latent_dim = latent_dim
        self.cfm = CFMDiT(latent_dim=latent_dim, cond_dim=self._cond_dim).to(self.device).eval()
        if self.cfg.cfm_ckpt and os.path.isfile(self.cfg.cfm_ckpt):
            sd = torch.load(self.cfg.cfm_ckpt, map_location=self.device)
            self.cfm.load_state_dict(sd)
            print(f"[cfm] Loaded trained CFM-DiT weights: {self.cfg.cfm_ckpt}")
        else:
            print("[cfm] !!! UNTRAINED !!! CFM-DiT is randomly initialized. "
                  "The pipeline will run end-to-end, but the *generated* audio "
                  "will be structured noise until you train/fine-tune this core "
                  "(see the training-stub notes at the bottom of the file).")
        return self

    # ---- Codec helpers ------------------------------------------------------
    @torch.no_grad()
    def encode_reference(self, reference_audio_path: str):
        """
        Step A: raw reference wav -> WavTokenizer continuous latents + discrete
        single-layer tokens (timbre / rhythm carrier).
        Returns (features [1, C, T], discrete_codes).
        """
        try:
            from encoder.utils import convert_audio
        except ImportError:
            # fallback resampler if the repo helper isn't importable
            def convert_audio(w, sr, target_sr, target_ch):
                if w.size(0) > target_ch:
                    w = w.mean(0, keepdim=True)
                if sr != target_sr:
                    w = torchaudio.functional.resample(w, sr, target_sr)
                return w

        wav, sr = torchaudio.load(reference_audio_path)
        wav = convert_audio(wav, sr, WT_SAMPLE_RATE, 1).to(self.device)
        bandwidth_id = torch.tensor([0], device=self.device)
        # WavTokenizer: single quantizer -> features [1, C, T], codes [1, 1, T]
        features, discrete_code = self.wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        n_tok = discrete_code.shape[-1]
        print(f"[encode] reference -> {n_tok} tokens "
              f"(~{n_tok / (wav.shape[-1] / WT_SAMPLE_RATE):.0f} tok/s), "
              f"latent shape {tuple(features.shape)}")
        return features, discrete_code

    @torch.no_grad()
    def decode_latents(self, features: torch.Tensor) -> torch.Tensor:
        """Step D: continuous latents [1, C, T] -> waveform [1, N] @ 24 kHz."""
        bandwidth_id = torch.tensor([0], device=self.device)
        audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        return audio.detach().cpu()

    # ---- 4d. THE MAIN GENERATION FUNCTION -----------------------------------
    @torch.no_grad()
    def generate_from_prompt_and_reference(
        self,
        text_prompt: str,
        reference_audio_path: str,
        duration_seconds: float = 5.0,
        steps: int = 6,
        out_path: str = "sota_output_sample.wav",
    ) -> str:
        assert self.wavtokenizer is not None, "call load_codec() first"
        assert self.text_model is not None, "call load_text_encoder() first"

        # --- Step A: reference encoding (timbre/rhythm in-context latents) ---
        ref_feats, _ = self.encode_reference(reference_audio_path)   # [1, C, T]
        latent_dim = ref_feats.shape[1]
        if self.cfm is None:
            self.load_cfm(latent_dim=latent_dim)
        ref_latent = ref_feats.transpose(1, 2)                       # [1, T, C]

        # --- Step B: text embedding ------------------------------------------
        cond_vec = self.embed_text(text_prompt)                     # [1, cond_dim]

        # --- Step C: CFM generation (solve the ODE in 4-8 steps) -------------
        gen_len = max(1, int(round(duration_seconds * self.cfg.tokens_per_second)))
        print(f"[cfm] generating {gen_len} latent frames "
              f"(~{duration_seconds:.1f}s) in {steps} ODE steps")
        gen_latent = self.cfm.sample(cond_vec, ref_latent, gen_len=gen_len, steps=steps)
        gen_feats = gen_latent.transpose(1, 2).contiguous()         # [1, C, gen_len]

        # --- Step D: decode to high-fidelity waveform ------------------------
        audio = self.decode_latents(gen_feats)                      # [1, N]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(out_path, audio, WT_SAMPLE_RATE)
        print(f"[done] wrote {out_path}")
        return out_path

    # ---- Bonus: pure codec round-trip (fully working, no training needed) ---
    @torch.no_grad()
    def reconstruct(self, reference_audio_path: str, out_path: str = "sota_output_sample.wav") -> str:
        """Encode -> decode a real reference. Verifies your WavTokenizer setup."""
        feats, _ = self.encode_reference(reference_audio_path)
        audio = self.decode_latents(feats)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(out_path, audio, WT_SAMPLE_RATE)
        print(f"[done] reconstruction -> {out_path}")
        return out_path


# =============================================================================
#  5. CLI + optional GRADIO UI
# =============================================================================
def build_pipeline(args) -> SOTAAudioGenPipeline:
    if args.wavtokenizer_repo:
        sys.path.insert(0, os.path.abspath(args.wavtokenizer_repo))
    cfg = PipelineConfig(
        wt_config=args.wt_config,
        wt_ckpt=args.wt_ckpt,
        text_model=args.text_model,
        cfm_ckpt=args.cfm_ckpt,
        tokens_per_second=args.tokens_per_second,
    )
    pipe = SOTAAudioGenPipeline(cfg)
    pipe.load_codec()
    if args.mode == "generate":
        pipe.load_text_encoder()
    return pipe


def launch_gradio(pipe: SOTAAudioGenPipeline):
    import gradio as gr

    def _run(prompt, ref, dur, steps):
        return pipe.generate_from_prompt_and_reference(
            prompt, ref, duration_seconds=float(dur), steps=int(steps),
            out_path="sota_output_sample.wav",
        )

    with gr.Blocks(title="SOTA Audio Gen (WavTokenizer + CFM-DiT)") as demo:
        gr.Markdown("## WavTokenizer + CFM-DiT — local MVP")
        prompt = gr.Textbox(label="Text prompt", value="warm analog synth pad, slow attack")
        ref = gr.Audio(label="Reference .wav (style/timbre)", type="filepath")
        dur = gr.Slider(1, 15, value=5, step=1, label="Duration (s)")
        steps = gr.Slider(4, 8, value=6, step=1, label="ODE steps")
        btn = gr.Button("Generate")
        out = gr.Audio(label="Output", type="filepath")
        btn.click(_run, [prompt, ref, dur, steps], out)
    demo.launch()


def main():
    p = argparse.ArgumentParser(description="SOTA Text-to-Audio (WavTokenizer + CFM-DiT) MVP")
    p.add_argument("--mode", choices=["generate", "reconstruct", "gradio"], default="generate")
    p.add_argument("--wavtokenizer-repo", default=None,
                   help="path to cloned jishengpeng/WavTokenizer (if not running from inside it)")
    p.add_argument("--wt-config", required=True, help="WavTokenizer .yaml config")
    p.add_argument("--wt-ckpt", required=True, help="WavTokenizer .ckpt weights")
    p.add_argument("--text-model", default="laion/clap-htsat-unfused",
                   help="CLAP id (default) or a T5 encoder id, e.g. google/t5-v1_1-base")
    p.add_argument("--cfm-ckpt", default=None, help="trained CFM-DiT weights (.pt); omit for template mode")
    p.add_argument("--tokens-per-second", type=int, default=75)
    p.add_argument("--prompt", default="warm analog synth pad, slow attack")
    p.add_argument("--reference", default=None, help="path to reference .wav")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--out", default="sota_output_sample.wav")
    args = p.parse_args()

    pipe = build_pipeline(args)

    if args.mode == "gradio":
        pipe.load_text_encoder()
        launch_gradio(pipe)
    elif args.mode == "reconstruct":
        assert args.reference, "--reference required for reconstruct mode"
        pipe.reconstruct(args.reference, out_path=args.out)
    else:  # generate
        assert args.reference, "--reference required for generate mode"
        pipe.generate_from_prompt_and_reference(
            args.prompt, args.reference,
            duration_seconds=args.duration, steps=args.steps, out_path=args.out,
        )


if __name__ == "__main__":
    main()


# =============================================================================
#  6. TRAINING THE CFM-DiT  (why "template mode" exists + how to make it real)
# =============================================================================
#  Conditional Flow Matching objective (rectified-flow / straight paths):
#
#    Given clean latent x1 (from WavTokenizer.encode of a training clip) and
#    noise x0 ~ N(0, I), sample flow-time t ~ U(0,1) and form the straight path
#        x_t = (1 - t) * x0 + t * x1
#    The target velocity is constant along the path:
#        v_target = x1 - x0
#    Train the DiT to regress it (only on the generated / masked region):
#
#        v_pred = cfm(x_t, t, cond_vec, ref_latent)
#        loss   = F.mse_loss(v_pred[:, ref_len:], v_target[:, ref_len:])
#
#  At inference we integrate dx/dt = v_pred from t=0 (noise) to t=1 (data) with
#  a 4-8 step Euler solver — exactly what CFMDiT.sample() does above. Because
#  the paths are straight, few steps suffice; this is the "SOTA fast" regime.
#
#  You need a dataset of (audio clip, text caption) pairs. Encode audio with
#  WavTokenizer to get x1; use a slice as the in-context ref, predict the rest.
#
#  ---------------------------------------------------------------------------
#  DON'T want to train? Use a trained CFM-DiT that already exists:
#    * F5-TTS  (SWivid/F5-TTS): CFM DiT + in-context reference, 4-32 NFE.
#      It runs in mel space with a Vocos vocoder (not WavTokenizer), but it is
#      the closest downloadable realization of the exact paradigm you described
#      and produces real audio today. Swap load_codec()+decode for its
#      mel<->vocos path and reuse this file's sampler/CLI structure.
#    * InspireMusic / Stable Audio Open: trained text->audio latent-diffusion,
#      good if you want music/SFX rather than speech-style in-context cloning.
# =============================================================================
