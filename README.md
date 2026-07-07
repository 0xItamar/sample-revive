# Audio Restoration & Super-Resolution Benchmark

A modular Python pipeline to A/B test open-source audio restoration /
super-resolution models against a classical DSP baseline, and visually inspect
recovered low-end / high-end via spectrograms.

Take a **bad sample** (low sample rate, missing bands, background noise), push it
through several models, and compare outputs + spectrograms.

---

## Also in this repo: AI sample generation (Stable Audio 3 + CLAP)

A separate pipeline to **generate drum/synth one-shots from text** with Stability
AI's open-weight **Stable Audio 3**, and **fine-tune it (LoRA)** on your own
sample library — auto-captioned with **CLAP**.

| Module | What it is | Docs |
|---|---|---|
| [`clap/`](clap/README.md) | CLAP audio→text tagging API; auto-captions a sample folder | [clap/README.md](clap/README.md) |
| [`stable/`](stable/README.md) | Stable Audio 3 generation API **+ LoRA fine-tuning** | [stable/README.md](stable/README.md) |

Quickstart (in `.venv-sa3`, Python 3.11 — see [stable/README.md](stable/README.md)):

```bash
# 1. caption your samples with CLAP  ->  clap/captions.csv
python -m clap.api --src ~/Music/Kicks --out clap/captions.csv

# 2. generate one-shots from text (base model)
python -m stable.api --prompt "deep 808 sub kick, single hit, dry" --n 8 --out out/

# 3. LoRA fine-tune on your library, then generate in its character
python -m stable.train --captions clap/captions.csv --src ~/Music/Kicks \
    --out stable/checkpoints/kicks_lora.safetensors
python -m stable.api --prompt "punchy kick, single hit, dry" \
    --lora stable/checkpoints/kicks_lora.safetensors --out out/
```

Committed examples: the **CLAP mapping** in [`clap/captions.csv`](clap/captions.csv)
(+ readable `captions.txt`), and seed-matched **before/after LoRA** outputs in
[`stable/outputs/`](stable/outputs/).

---

## What's included

| Entry | Type | What it does |
|-------|------|--------------|
| `BaselineEnhancer` | Classical DSP (control) | `noisereduce` spectral gating + scipy band-pass + normalise. **Cannot invent missing bands** — that's the control. |
| `AudioSRModel` | Latent diffusion | [HaoheLiu/AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) — generative 48 kHz super-resolution. |
| `TemplateModel` | Stub | Copy/fill to plug in LavaSR, a Flow-Matching restorer, VoiceFixer, etc. |

Everything implements one interface (`RestorationModel.process`), so adding a
model auto-joins the benchmark **and** the spectrogram grid.

## Run it in Google Colab (recommended, GPU)

1. **New notebook → Runtime → Change runtime type → GPU (T4 is fine).**
2. Get the code into the session (either clone this repo or upload the `.py`):
   ```python
   !git clone https://github.com/0xItamar/sample-revive.git
   %cd sample-revive
   ```
3. **Cell 1 — install deps** (run once):
   ```python
   from audio_restoration_benchmark import install_dependencies
   install_dependencies()
   ```
   If pip changes `torch`/`numpy`, do **Runtime → Restart session**, then continue
   *without* re-running the install.
4. **Cell 2 — upload your input** as `input_bad_sample.wav`:
   ```python
   from google.colab import files
   files.upload()   # pick your degraded wav; or skip to use a synthetic one
   ```
5. **Cell 3 — run the whole thing:**
   ```python
   from audio_restoration_benchmark import main
   results = main()
   ```
   `main()` will generate a synthetic bad sample automatically if none is present,
   run every model, print a summary + spectral stats, and render the spectrogram
   comparison.
6. **Cell 4 — listen / download:**
   ```python
   import IPython.display as ipd
   for r in results:
       if r.ok:
           print(r.name); ipd.display(ipd.Audio(r.output_path))
   ```

> **Notebook structure tip:** the `.py` file is written with `# %%` cell markers.
> In Colab you don't paste the whole file — import the functions you need per cell
> as shown above. Keep *install*, *upload*, and *run* in separate cells so a
> restart never forces a re-install.

## Run it locally

```bash
python -c "from audio_restoration_benchmark import install_dependencies as f; f()"
python audio_restoration_benchmark.py       # generates a synthetic sample if none
```

## Outputs

- `outputs/output_baseline.wav`, `outputs/output_audiosr.wav`, …
- `comparison_spectrograms.png` — log-frequency spectrograms, input vs each output.
- A console table of reference-free spectral descriptors (centroid / bandwidth /
  roll-off) — higher generally means more restored top-end.

## Adding a model (e.g. LavaSR / Flow-Matching)

1. Copy `TemplateModel` in Cell 6.
2. Set `name` (drives `output_<name>.wav`), implement `setup()` (load weights once)
   and `process(input_path, output_path)` (write a 48 kHz float WAV).
3. Add an instance to the `MODELS` list in Cell 7. Done — it's in the bench.

## Gotchas

- **AudioSR checkpoints** (~2 GB) download to `~/.cache/audiosr` on first run.
- `audiosr` pins `diffusers`/`transformers`/`torchaudio`; if imports break after
  install, **restart the runtime** and don't re-run the install cell.
- Evaluation here is **reference-free** (you rarely have a clean reference for a
  real bad sample), so it's primarily visual. If you *do* have a clean reference,
  add LSD / SI-SDR against it for hard numbers.
