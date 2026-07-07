# Stable Audio 3 API (generation + LoRA fine-tuning)

Text → audio one-shots with Stability AI's open-weight **Stable Audio 3**, plus
LoRA fine-tuning to imprint your own sample library's character.

## What is Stable Audio 3?

An open-weight **text-to-audio latent diffusion** model. A prompt is encoded by
a text encoder (**t5gemma**) into embedding vectors that condition a **diffusion
transformer (DiT)**; the DiT denoises a latent, and a **VAE** decodes it to
44.1 kHz stereo audio.

```
prompt ──t5gemma──► text embeddings ──┐
                                       ├──► DiT (denoise latent) ──► VAE ──► audio
                    noise / masked in ─┘
```

We use **`small-sfx`** (0.6B) — the sound-effects-tuned checkpoint, best for
drums/one-shots. It is a `diffusion_cond_inpaint` model (see *Inpainting* below).

### Model choices
| checkpoint | size | use |
|---|---|---|
| `small-sfx` | 0.6B | **drums / SFX one-shots** (what this API defaults to) |
| `small-music` | 0.6B | melodic instruments |
| `medium` | 2B | biggest downloadable; general music + SFX / instruments |

`small-sfx` **cannot** do pitched instruments (flute etc.) — use `medium`/`small-music`.

## Generation API

```python
from stable.api import StableAudio

sa = StableAudio("models/stable-audio-3-small-sfx")
sa.generate_oneshots("deep 808 sub kick, single hit, dry", n=6, out_dir="out")

# with a trained LoRA:
sa = StableAudio("models/stable-audio-3-small-sfx",
                 lora="stable/checkpoints/kicks_lora.safetensors", lora_strength=1.0)
sa.generate("punchy kick, single hit, dry", seconds=1.0, steps=8, cfg=1.5, seed=0)
```

CLI:

```bash
python -m stable.api --prompt "punchy kick, single hit, dry" --n 8 --out out/
python -m stable.api --prompt "..." --lora stable/checkpoints/kicks_lora.safetensors --lora-strength 0.7
```

## Harnessed generation

Use the harness when you need repeatable runs with prompt/model/seed settings
logged next to the generated audio.

```bash
python -m stable.harness --config stable/configs/medium_kick_baseline.json --limit 1
```

The default medium config expects local weights at
`models/stable-audio-3-medium`. Override that path for a machine-specific cache:

```bash
python -m stable.harness \
    --config stable/configs/medium_kick_baseline.json \
    --weights /path/to/stable-audio-3-medium \
    --limit 1
```

Harness outputs are written under `stable/runs/` and ignored by Git:

- `manifest.json` — full config plus per-output metadata
- `runs.jsonl` — one row per generation
- `audio/*.wav` — generated files

### `StableAudio`
| method | description |
|---|---|
| `StableAudio(weights_dir, device=None, lora=None, lora_strength=None)` | Load base (local dir or hub); optionally apply a LoRA. |
| `.generate(prompt, negative, seconds, steps, cfg, seed)` | One clip → `(channels, samples)` peak-normalized CPU tensor. |
| `.generate_oneshots(prompt, n, out_dir, ..., trim, floor_ratio)` | N variations, transient-trimmed, saved as WAV. |
| `trim_transient(wav, sr, floor_ratio=0.12)` | Windowed-RMS trim to the hit (module function). |

### Settings that matter (small-sfx, learned empirically)
- **Low guidance:** `cfg` 1–2. `cfg ≥ 4` over-drives into dense saturation.
- **It fills the duration:** short `seconds ≈ 1.0` → single hit; long → sustained texture.
- **Describe the character:** `"deep 808 sub kick, single hit, dry"` gives a clean
  decaying hit; bare `"kick drum"` gives a sustained drone.
- Base (`*-base`) checkpoints want `cfg ≈ 7`; the tuned checkpoints want `cfg ≈ 1`.

## Inpainting (why training needs a mask)

`small-sfx` can regenerate a **masked region** of audio while keeping the rest —
audio inpainting. Its forward pass therefore always expects `inpaint_mask` +
`inpaint_masked_input`. **Plain text-to-audio is the special case where the whole
clip is masked** ("generate everything"). This is *one model* fed different
inputs — the mask is data (0/1 over the timeline), not a separate model.

## LoRA fine-tuning

Trains a small **LoRA adapter** (~10M params) while the 0.6B base + t5gemma stay
**frozen** — feasible on a Mac (MPS), ~1.5–3 s/step.

```bash
python -m stable.train --steps 1500 --batch 2 \
    --captions clap/captions.csv --src ~/Music/Kicks \
    --out stable/checkpoints/kicks_lora.safetensors
```

Or from Python: `from stable.train import train_lora`.

Key points (see `train.py` docstring for detail):
- Captions come from **`clap/captions.csv`** (filename + CLAP descriptors).
- **`inpainting_config={"mask_kwargs":{}}` + `p_one_shot=0.5` are required** — the
  first supplies the mask conditioning small-sfx demands (else `KeyError:
  'inpaint_mask'`), the second trains full-generation ~half the time so the LoRA
  learns to generate complete kicks, not just inpaint.
- **1500–3000 steps** (5–10 epochs of 552 kicks) is the sweet spot; more risks
  overfitting. **rank 16** default; 32 for a stronger imprint.
- Apply the result via `StableAudio(..., lora=...)`; blend with `lora_strength`.

## Outputs (committed)

Seed-matched A/B (prompt `kick drum, deep 808 sub, punchy, single hit, dry`):

- **`outputs/before_lora/`** — base `small-sfx` (tighter, ~0.35 s avg)
- **`outputs/after_lora/`** — with `kicks_lora` (longer with more body/decay,
  ~0.62 s avg — the training library's character)

`checkpoints/kicks_lora.safetensors` is the trained rank-16 adapter (20 MB).
