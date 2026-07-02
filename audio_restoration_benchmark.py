"""
Audio Restoration & Super-Resolution Benchmarking Pipeline
==========================================================

A modular pipeline to A/B test open-source audio restoration / super-resolution
models against a classical DSP baseline.

Designed to run in Google Colab (GPU runtime) but also works locally.

The file is written as a set of `# %%` cells so you can either:
  - Run it top-to-bottom as a script:      python audio_restoration_benchmark.py
  - Or split it into Colab / VS Code cells (each `# %%` marker is one cell).

Author's note (Senior Audio AI Engineer):
  Classical DSP (baseline) CANNOT invent missing frequency content — it can only
  clean up what already exists. Generative models (AudioSR) hallucinate plausible
  high/low bands from a learned prior. The point of this bench is to *see* that
  difference in the spectrograms, not just hear it.
"""

# %% [markdown]
# ## Cell 1 — Environment setup
# Run this ONCE per Colab session. After it finishes, if pip changed torch/numpy
# versions, use the Colab menu: Runtime -> Restart session, then skip this cell.

# %%
def install_dependencies():
    """Install everything the pipeline needs.

    Notes for Colab:
      * `audiosr` pulls in diffusers/transformers/torchaudio at pinned versions.
        It can downgrade the pre-installed torch — if you hit an import error,
        restart the runtime and DO NOT re-run this cell.
      * First AudioSR run downloads ~2GB of checkpoints to ~/.cache/audiosr.
    """
    import subprocess
    import sys

    pkgs = [
        # --- core DL stack (usually already in Colab, listed for completeness) ---
        "torch",
        "torchaudio",
        # --- generative SR model + its ecosystem ---
        "audiosr",            # HaoheLiu/versatile_audio_super_resolution
        "diffusers",
        "transformers",
        # --- classical baseline + DSP / IO ---
        "noisereduce",
        "librosa",
        "soundfile",
        "scipy",
        # --- viz ---
        "matplotlib",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        check=False,  # keep going even if one optional pkg fails to resolve
    )
    print("Dependency install attempted. If torch/numpy changed, RESTART the runtime.")


# In a notebook you'd just call install_dependencies() in the first cell.
# Guarded so importing this module for its classes does not trigger an install.
if __name__ == "__main__" and False:  # flip to True the first time you run in Colab
    install_dependencies()


# %% [markdown]
# ## Cell 2 — Imports & global config

# %%
import os
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

# librosa is heavy; import lazily where possible, but we use it a lot so import now.
import librosa

# ------------------------- Global configuration ---------------------------------
INPUT_FILE = "input_bad_sample.wav"     # your degraded input
TARGET_SR = 48000                        # AudioSR outputs 48 kHz; we align everyone to it
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %% [markdown]
# ## Cell 3 — Core abstractions & audio I/O helpers
# Every model implements the same tiny interface, so the runner and the plotting
# code never need to know which model produced a file.

# %%
def load_audio(path, sr=None, mono=True):
    """Load audio as float32 numpy in [-1, 1]. Returns (wav, sr)."""
    wav, file_sr = librosa.load(path, sr=sr, mono=mono)
    return wav.astype(np.float32), file_sr


def save_audio(path, wav, sr):
    """Save a 1-D (or (samples, channels)) float array to WAV."""
    wav = np.asarray(wav).squeeze()
    # soundfile wants (samples,) or (samples, channels); guard against (ch, samples)
    if wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
        wav = wav.T
    # clip to avoid integer-wrap artefacts on export
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(path, wav, sr)
    return path


@dataclass
class RunResult:
    """One model's output + bookkeeping for the report."""
    name: str
    output_path: str = ""
    sr: int = 0
    duration_s: float = 0.0
    runtime_s: float = 0.0
    ok: bool = False
    error: str = ""


class RestorationModel(ABC):
    """Common interface for every entry in the benchmark.

    Subclass this to add a new model. Implement `process()`; the runner handles
    timing, error isolation, and reporting for you.
    """

    #: Human-readable name; also used in the output filename (output_<slug>.wav)
    name = "abstract"

    def setup(self):
        """Optional: load weights / build the model once before processing."""
        pass

    @abstractmethod
    def process(self, input_path: str, output_path: str) -> str:
        """Read `input_path`, restore it, write to `output_path`, return the path."""
        raise NotImplementedError


# %% [markdown]
# ## Cell 4 — Baseline (control group): classical DSP enhancement
# Denoise (spectral gating via `noisereduce`) + a gentle band-pass to strip
# rumble/hiss + loudness normalisation, then resample to the common target SR.
# This CANNOT recover missing high-end — that's exactly the point of the control.

# %%
from scipy.signal import butter, sosfilt


def _bandpass(wav, sr, low_hz=80.0, high_hz=None, order=4):
    """Zero-rumble / anti-hiss band-pass. high_hz defaults to just below Nyquist."""
    nyq = sr / 2.0
    high_hz = high_hz if high_hz is not None else nyq * 0.98
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, wav).astype(np.float32)


class BaselineEnhancer(RestorationModel):
    name = "baseline"

    def __init__(self, denoise=True, prop_decrease=0.9):
        self.denoise = denoise
        self.prop_decrease = prop_decrease

    def process(self, input_path, output_path):
        wav, sr = load_audio(input_path, sr=None, mono=True)

        # 1) Spectral-gating noise reduction (estimates noise from the signal itself)
        if self.denoise:
            import noisereduce as nr
            wav = nr.reduce_noise(
                y=wav, sr=sr, stationary=False, prop_decrease=self.prop_decrease
            ).astype(np.float32)

        # 2) Clean up sub-sonic rumble and near-Nyquist junk
        wav = _bandpass(wav, sr, low_hz=80.0)

        # 3) Peak-normalise to -1 dBFS
        peak = np.max(np.abs(wav)) + 1e-9
        wav = 0.891 * wav / peak  # 0.891 ~= -1 dBFS

        # 4) Align sample rate to the rest of the bench (no new bands are created)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        return save_audio(output_path, wav, sr)


# %% [markdown]
# ## Cell 5 — AudioSR (HaoheLiu): latent-diffusion super-resolution
# Uses the official `audiosr` package. It ingests audio at any SR and generates
# a 48 kHz, full-band restoration by denoising in a latent space conditioned on
# the input. `ddim_steps` and `guidance_scale` are your main quality/speed knobs.

# %%
class AudioSRModel(RestorationModel):
    name = "audiosr"

    def __init__(
        self,
        model_name="basic",      # "basic" (general audio) or "speech"
        ddim_steps=50,           # fewer = faster, more = cleaner (try 25–100)
        guidance_scale=3.5,      # classifier-free guidance strength
        seed=42,
        latent_t_per_second=12.8,
    ):
        self.model_name = model_name
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.latent_t_per_second = latent_t_per_second
        self._model = None

    def setup(self):
        from audiosr import build_model
        # device="auto" lets audiosr pick cuda if available
        self._model = build_model(model_name=self.model_name, device="auto")

    def process(self, input_path, output_path):
        from audiosr import super_resolution

        if self._model is None:
            self.setup()

        waveform = super_resolution(
            self._model,
            input_path,
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            ddim_steps=self.ddim_steps,
            latent_t_per_second=self.latent_t_per_second,
        )
        # audiosr returns shape like (1, 1, N) or (1, N); save_audio squeezes it.
        # Output SR is fixed to 48 kHz by the model.
        return save_audio(output_path, waveform, 48000)


# %% [markdown]
# ## Cell 6 — Plug-in template for the NEXT model (LavaSR / Flow-Matching / etc.)
# Copy this stub, fill in `setup()` and `process()`, add it to the MODELS list,
# and it automatically joins the benchmark + spectrogram comparison.

# %%
class TemplateModel(RestorationModel):
    """STUB — replace the body to integrate a new model.

    Example targets:
      * LavaSR                     (audio super-resolution)
      * A Flow-Matching restorer   (e.g. a FlowMatch / rectified-flow vocoder)
      * VoiceFixer, Nvidia BigVGAN, Resemble-Enhance, etc.

    Implementation checklist:
      1. In setup(): load weights / build the pipeline ONCE (cached in self._model).
      2. In process(): read input_path -> run model -> write 48 kHz float WAV.
      3. Add an instance to the MODELS list in Cell 7.
    """

    name = "template"  # change to e.g. "lavasr"; drives output_<name>.wav

    def __init__(self, **kwargs):
        self.cfg = kwargs
        self._model = None

    def setup(self):
        # e.g.
        #   from lavasr import load_model
        #   self._model = load_model(**self.cfg).to(DEVICE)
        raise NotImplementedError("Fill in setup() for your model.")

    def process(self, input_path, output_path):
        # Typical shape:
        #   wav, sr = load_audio(input_path, sr=<model_expected_sr>)
        #   out = self._model(wav)                    # -> numpy/torch tensor
        #   out = out.detach().cpu().numpy()
        #   return save_audio(output_path, out, TARGET_SR)
        raise NotImplementedError("Fill in process() for your model.")


# %% [markdown]
# ## Cell 7 — Pipeline runner
# Iterates over every configured model, isolates failures (one broken model does
# not abort the run), and prints a compact report.

# %%
def slugify(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower())


def run_benchmark(models, input_file=INPUT_FILE):
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"'{input_file}' not found. Upload it, or run make_synthetic_bad_sample()."
        )

    results = []
    for model in models:
        res = RunResult(name=model.name)
        out_path = os.path.join(OUTPUT_DIR, f"output_{slugify(model.name)}.wav")
        print(f"\n=== Running: {model.name} ===")
        t0 = time.time()
        try:
            model.setup()
            model.process(input_file, out_path)
            wav, sr = load_audio(out_path, sr=None, mono=True)
            res.output_path = out_path
            res.sr = sr
            res.duration_s = len(wav) / sr
            res.ok = True
            print(f"  -> saved {out_path}  ({sr} Hz, {res.duration_s:.1f}s)")
        except Exception as e:  # isolate per-model failure
            res.error = f"{type(e).__name__}: {e}"
            print(f"  !! FAILED: {res.error}")
            traceback.print_exc()
        res.runtime_s = time.time() - t0
        results.append(res)

    # ------- summary report -------
    print("\n" + "=" * 60)
    print(f"{'MODEL':<14}{'STATUS':<10}{'SR':<8}{'RUNTIME':<10}")
    print("-" * 60)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"{r.name:<14}{status:<10}{str(r.sr):<8}{r.runtime_s:>6.1f}s")
    print("=" * 60)
    return results


# ------------------------- CONFIGURE THE BENCHMARK HERE -------------------------
MODELS = [
    BaselineEnhancer(),
    AudioSRModel(ddim_steps=50, guidance_scale=3.5),
    # TemplateModel(),          # <- uncomment once you implement it
]


# %% [markdown]
# ## Cell 8 — Evaluation: spectrogram comparison
# Log-frequency spectrograms of the input vs every successful output, side by
# side. This is where you visually confirm recovered low-end / high-end.

# %%
def plot_spectrograms(input_file, results, out_png="comparison_spectrograms.png"):
    import matplotlib.pyplot as plt
    import librosa.display

    panels = [("input (bad sample)", input_file)]
    panels += [(r.name, r.output_path) for r in results if r.ok]

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    img = None
    for ax, (title, path) in zip(axes, panels):
        wav, sr = load_audio(path, sr=None, mono=True)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(wav, n_fft=2048, hop_length=512)), ref=np.max
        )
        img = librosa.display.specshow(
            D, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax, cmap="magma"
        )
        ax.set_title(f"{title}  —  {sr} Hz")
        ax.set_ylabel("Hz")

    if img is not None:
        fig.colorbar(img, ax=axes, format="%+2.0f dB")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved {out_png}")
    plt.show()


def quick_spectral_stats(results):
    """Reference-free descriptors: higher centroid/bandwidth => more restored top-end."""
    print(f"\n{'MODEL':<14}{'centroid Hz':<14}{'bandwidth Hz':<14}{'rolloff95 Hz':<14}")
    print("-" * 56)
    for r in results:
        if not r.ok:
            continue
        wav, sr = load_audio(r.output_path, sr=None, mono=True)
        cen = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))
        bw = float(np.mean(librosa.feature.spectral_bandwidth(y=wav, sr=sr)))
        roll = float(np.mean(librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.95)))
        print(f"{r.name:<14}{cen:<14.0f}{bw:<14.0f}{roll:<14.0f}")


# %% [markdown]
# ## Cell 9 — (Handy) Make a synthetic "bad sample" if you don't have one
# Degrades a clean signal so the whole notebook runs end-to-end. Replace with
# your own real `input_bad_sample.wav` for a meaningful comparison.

# %%
def make_synthetic_bad_sample(path=INPUT_FILE, seconds=4.0, clean_sr=44100, bad_sr=8000):
    """Generate -> degrade (down-sample + band-limit + noise) -> save at low SR."""
    from scipy.signal import resample_poly

    t = np.linspace(0, seconds, int(clean_sr * seconds), endpoint=False)
    # a couple of harmonics + a high partial so we can SEE the top-end get killed
    clean = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 6000 * t)
    ).astype(np.float32)

    # degrade: down-sample to bad_sr (kills everything above bad_sr/2 = 4 kHz) ...
    down = resample_poly(clean, bad_sr, clean_sr)
    # ... add broadband noise (deterministic seed for reproducibility)
    rng = np.random.default_rng(0)
    down = down + 0.02 * rng.standard_normal(len(down)).astype(np.float32)

    save_audio(path, down.astype(np.float32), bad_sr)
    print(f"Wrote synthetic bad sample -> {path} ({bad_sr} Hz)")
    return path


# %% [markdown]
# ## Cell 10 — Main entry point
# In Colab, call these one after another in separate cells instead of __main__.

# %%
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"'{INPUT_FILE}' missing — generating a synthetic one for the demo.")
        make_synthetic_bad_sample()

    results = run_benchmark(MODELS, INPUT_FILE)
    quick_spectral_stats(results)
    plot_spectrograms(INPUT_FILE, results)
    return results


if __name__ == "__main__":
    main()
