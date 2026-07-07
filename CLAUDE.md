# CLAUDE.md — Stable Audio 3 + CLAP sample-generation & LoRA runbook

Operational guide for the AI sample-generation pipeline in this repo (`clap/`,
`stable/`). Written so a fresh session can **train a new LoRA on a different
sample set** and generate one-shots without re-deriving anything.

> Sibling project in this repo: `audio_restoration_benchmark.py` (MusicGen→AudioSR
> restoration bench) — unrelated; different venvs. See root `README.md`.

---

## 0. TL;DR — run a new LoRA on a new folder

```bash
# always use the SA3 venv (Python 3.11)
V=.venv-sa3/bin/python

# 1. (if Ableton .aif) export the samples to WAV first — see §4. Then:
# 2. CLAP-caption the folder  ->  a captions.csv
$V -m clap.api --src ~/path/to/NewSamples --out clap/newsamples.csv --topk 4

# 3. LoRA fine-tune small-sfx on them (~35-40 min on MPS for 1500 steps)
$V -m stable.train \
    --captions clap/newsamples.csv --src ~/path/to/NewSamples \
    --out stable/checkpoints/newsamples_lora.safetensors \
    --steps 1500 --batch 2 --rank 16

# 4. Generate with the new LoRA
$V -m stable.api --prompt "deep punchy hit, single hit, dry" --n 8 \
    --lora stable/checkpoints/newsamples_lora.safetensors --out out/newsamples
```

That's the whole loop. Details, knobs, and gotchas below.

---

## 1. Environment (`.venv-sa3`, Python 3.11, MPS)

Everything runs in **`.venv-sa3`** (gitignored). If it's missing, rebuild it:

```bash
python3.11 -m venv .venv-sa3
.venv-sa3/bin/pip install --upgrade pip
# core: SA3 is NOT on PyPI — install from Stability's git (bundles SA3 model code)
.venv-sa3/bin/pip install "git+https://github.com/Stability-AI/stable-audio-3.git"
.venv-sa3/bin/pip install huggingface_hub laion_clap pytorch_lightning dill einops
```

**Exact working pins** (installing SA3 pulls most; these are the ones that MUST be
right — see §6 for why):

| package | version | note |
|---|---|---|
| python | 3.11 | audio ML libs lag on 3.13 |
| torch | 2.7.1 | pinned by stable-audio-3 |
| torchvision | **0.22.1** | MUST match torch 2.7.1 (else `torchvision::nms` error) |
| transformers | 5.13.0 | needs `T5GemmaEncoderModel` |
| numpy | **1.26.4** | NOT 2.x — pandas/sklearn ABI break |
| pandas | 2.1.4 | numpy-1.x-era wheel |
| scikit-learn | 1.3.2 | numpy-1.x-era wheel |
| pytorch_lightning | 2.1.0 | training loop |
| dill | (any) | dataset serializes the metadata fn |
| laion_clap | 1.1.4 | CLAP tagging |

Ignore pip's metadata-conflict *warnings* — runtime ABI is what matters, and the
above combination imports and runs. Device is **MPS** (Apple Silicon); training
and inference both work on it.

---

## 2. Models (gated HF weights, under `models/`, gitignored)

Weights live in `models/` (multi-GB, never committed). They're gated on Hugging
Face — you must be logged in.

```bash
# one-time auth (do in a REAL terminal — interactive token paste):
.venv-sa3/bin/hf auth login          # paste a read token from hf.co/settings/tokens
.venv-sa3/bin/hf auth whoami         # should print your username (was: itamard7)

# download a checkpoint (accept its license on the HF page first):
.venv-sa3/bin/hf download stabilityai/stable-audio-3-small-sfx \
    --local-dir models/stable-audio-3-small-sfx
```

| checkpoint | size | for | notes |
|---|---|---|---|
| `stable-audio-3-small-sfx` | 0.6B / 2.1 GB | **drums / SFX one-shots, kick LoRA** | default; already downloaded |
| `stable-audio-3-small-music` | 0.6B | melodic instruments | not yet downloaded |
| `stable-audio-3-medium` | 2B / 8.6 GB | biggest; general music + instruments | downloaded; use for flute/instruments |

**Model choice for a new LoRA:** train on the checkpoint that already does the
*class* of sound. Drums/percussion/SFX → `small-sfx`. Melodic/instrument →
`small-music` or `medium`. `small-sfx` **cannot** do pitched instruments.

Each SA3 repo also bundles its **t5gemma** text encoder in a subfolder (auto-used).
Only `small-sfx` is SFX-specialized; there is no bigger sfx model. `medium` is the
biggest downloadable; a `large` exists but is **API-only**.

---

## 3. The three modules

```
clap/api.py        ClapTagger — audio->text zero-shot tagging  ->  captions.csv
stable/api.py      StableAudio — text -> one-shots (+ --lora)
stable/train.py    train_lora() — LoRA fine-tune small-sfx on captioned audio
```

Library use:
```python
from clap.api import ClapTagger
from stable.api import StableAudio
from stable.train import train_lora
```

### CLAP tagging (`clap/api.py`)
Caption = `kick drum, {cleaned filename}, {top-k CLAP descriptors}, single hit, dry`.
Descriptor vocabulary is `ClapTagger.DEFAULT_TAGS` — **edit this list** (or pass
`tags=[...]`) to retarget for non-kick material (e.g. add "shimmer, metallic,
plucky, airy" for synths). The prompt template `f"a {t} kick drum"` is built in
`__init__` — change it if not tagging kicks.

### Generation (`stable/api.py`)
`StableAudio(weights_dir, lora=None, lora_strength=None).generate_oneshots(prompt, n, out_dir, seconds, steps, cfg, seed, trim, floor_ratio)`.

### Training (`stable/train.py`)
`train_lora(weights_dir, captions_csv, src_dir, out, steps, batch, lr, rank, sample_seconds)`.

---

## 4. Preparing a new sample folder

- **Format:** the trainer/CLAP read via ffmpeg/torchaudio → use **WAV (PCM), 44.1 kHz**.
  Mono or stereo both fine.
- **Ableton `.aif` are undecodable** — they use a proprietary `able` codec (AIFF-C)
  that ffmpeg/libsndfile/librosa CANNOT read (only Ableton Live can). If a folder
  fails to decode with "codec tag: able", export to WAV from Ableton first
  (Export Audio/Video → WAV, 44100, 24-bit). This is what happened with
  `~/Music/Kicks` (552 files).
- **Captions come from the filename + CLAP**, so descriptive filenames help. The
  filename's index numbers / pack cruft (`SR`, `MPL`, `Amp`) are stripped.
- **Dataset size:** a few hundred consistent samples is plenty for a useful LoRA.

---

## 5. Settings & knobs (what to change and why)

### Generation (small-sfx is tuned for LOW guidance)
- `--cfg` **1–2**. `cfg ≥ 4` over-drives into dense saturation (garbage). Base
  (`*-base`) checkpoints instead want `cfg ≈ 7`.
- `--seconds` **~1.0** for a single hit. small-sfx is an SFX model that **fills the
  requested duration** — a long duration turns a one-shot into a sustained texture.
  Use 2–4 s intentionally for pads/atmos.
- `--steps` 8–16 is plenty (it's a fast/distilled sampler).
- **Prompt style matters:** describe character ("deep 808 sub kick, single hit,
  dry") → clean decaying hit. Bare "kick drum" → sustained drone.
- `--floor` (0.12 default) tail-trim tightness: higher = tighter. `--no-trim` to keep full.
- `--lora-strength` 0..~1.5 blends the adapter (0 = base, 1 = full). No retrain needed to change it.

### Training
- `--steps` **1500–3000** (≈5–10 epochs of ~550 samples at batch 2). More risks
  **overfitting** a small library. Observed: loss 0.35 → 0.26 over 1500 steps.
- `--rank` **16** default; **32** for a stronger imprint (bigger adapter).
- `--batch` 2 (MPS memory); `--lr` 1e-4; `--sample-seconds` 3.0 (training crop; kicks are padded).
- Speed: **~1.5–3 s/step on MPS** → 1500 steps ≈ 35–40 min.
- **`inpainting_config` + `p_one_shot=0.5` are hard-coded and REQUIRED** in
  `train.py` — see §6. Don't remove them.

### Evaluating a LoRA
Always generate **seed-matched** base vs LoRA (same `--seed`, one with `--lora`):
```bash
$V -m stable.api --prompt "..." --n 6 --seed 300 --out out/base
$V -m stable.api --prompt "..." --n 6 --seed 300 --lora <lora> --out out/lora
```
Then listen. The kick LoRA shifted avg duration 0.35 s → 0.62 s (more body/decay)
— a measurable imprint of the training library.

---

## 6. Gotchas & fixes (already solved — don't rediscover)

**Install / versions**
- `stable_audio_3` is **not on PyPI** → install from git. The PyPI
  `stable-audio-tools 0.0.19` does NOT support SA3 (fails on `local_add_cond_dim`
  / t5gemma).
- `numpy` must be **1.26.4**, not 2.x → else `ValueError: numpy.dtype size changed`
  (pandas/sklearn ABI). Fix: `pip install "numpy==1.26.4" pandas==2.1.4 scikit-learn==1.3.2`.
- `torchvision` must match torch (0.22.1 ↔ 2.7.1) → else `operator torchvision::nms
  does not exist`.
- `transformers` must be ≥ 5 (have `T5GemmaEncoderModel`) → else model build fails.

**CLAP (`laion_clap`) — 3 patches, all in `clap/api.py`**
1. `torch.load` monkeypatched to `weights_only=False` (checkpoint has numpy globals).
2. `model.model.load_state_dict(strict=False)` (transformers ≥ 5 dropped RoBERTa
   `position_ids`).
3. audio passed as a torch tensor when `use_tensor=True`.
- CLAP loads a ~2 GB checkpoint on first `load_ckpt()` (cached after).

**Training**
- small-sfx is `diffusion_cond_inpaint` → forward pass needs `inpaint_mask` +
  `inpaint_masked_input`. `train.py` passes `inpainting_config={"mask_kwargs":{}}`
  so the training step builds them, and `p_one_shot=0.5` so ~half the steps train
  pure full-generation. Without it: `KeyError: 'inpaint_mask'`.
- SA3 has native LoRA (`stable_audio_3/models/lora/`, `training/diffusion.py` is a
  `pl.LightningModule`). No CLI train exists — `stable/train.py` is the harness.

**Inference model loading**
- `StableAudioModel.from_pretrained("small-sfx")` re-downloads to the HF cache. To
  use local `models/` weights, `stable/api.py` calls the internal
  `load_diffusion_cond(config, ckpt, device, model_half)` then wraps in
  `StableAudioModel(...)` — avoids the re-download.
- `sample_size` must be a multiple of the VAE downsampling ratio **4096**.

**Decoding**: Ableton `able`-codec `.aif` → export to WAV (see §4).

---

## 7. File map

```
clap/
  api.py            ClapTagger + CLI (python -m clap.api)
  README.md         CLAP explainer
  captions.csv      the kick mapping (552 rows: filepath,prompt)
  captions.txt      readable name -> caption
stable/
  api.py            StableAudio generation + CLI (python -m stable.api)
  train.py          LoRA training + CLI (python -m stable.train)
  README.md         SA3 / LoRA explainer
  checkpoints/
    kicks_lora.safetensors   trained rank-16 kick adapter (20 MB)
  outputs/
    before_lora/    6 base one-shots (seed 300)
    after_lora/     6 LoRA one-shots (seed 300, same prompt)
models/             gitignored: small-sfx (2.1G), medium (8.6G)
.venv-sa3/          gitignored: the Python 3.11 env
```

Merged to master in PR #2. The kick library is `~/Music/Kicks` (WAV, 552 files).
```
