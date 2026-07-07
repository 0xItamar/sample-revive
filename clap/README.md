# CLAP tagging API

Zero-shot acoustic tagging of audio samples, used to auto-caption a kick library
for Stable Audio 3 LoRA fine-tuning.

## What is CLAP?

**CLAP** (Contrastive Language-Audio Pretraining) is a joint **audio ↔ text**
embedding model. It has two encoders:

- an **audio tower** (HTSAT-style) that turns a clip into an embedding vector, and
- a **text tower** (**RoBERTa**) that turns a phrase into an embedding vector,

trained contrastively so a clip and its matching description land at **nearby
points in one shared space**. That enables **zero-shot tagging**: embed a clip,
embed candidate descriptor phrases, and rank descriptors by cosine similarity.

> CLAP's text encoder is **RoBERTa** — unrelated to Stable Audio 3's **t5gemma**.
> Both are text *encoders* (they output vectors), not text generators.

```
         embed audio ─┐
                      ├──► cosine similarity ──► top-k descriptors
 embed "punchy kick" ─┘
```

## API

```python
from clap.api import ClapTagger

tagger = ClapTagger(topk=4)            # loads the 630k checkpoint on first run
tagger.tag("kick.wav")                 # -> ['punchy', 'tight', 'sub bass', '808']
tagger.caption("kick.wav")             # -> 'kick drum, ..., punchy, tight, ..., single hit, dry'
tagger.tag_directory("~/Music/Kicks", "clap/captions.csv")
```

CLI:

```bash
python -m clap.api --src ~/Music/Kicks --out clap/captions.csv --topk 4
```

### `ClapTagger`
| method | description |
|---|---|
| `ClapTagger(tags=None, topk=4, device=None)` | Load CLAP; precompute vocabulary text embeddings. `tags` defaults to `DEFAULT_TAGS`. |
| `.embed_audio(path)` | L2-normalized CLAP embedding for one file. |
| `.tag(path)` | Top-k descriptor tags, most similar first. |
| `.caption(path)` | Training caption: cleaned filename + descriptors. |
| `.tag_directory(src, out_csv, limit=0)` | Tag a folder recursively → `filepath,prompt` CSV. |

The descriptor vocabulary (`DEFAULT_TAGS`): `808, sub bass, deep, boomy, punchy,
tight, snappy, clicky, distorted, saturated, gritty, clean, analog, acoustic,
synthetic, lo-fi, hard, soft, dark, bright, long tail, short`.

## Caption format

```
kick drum, {cleaned filename}, {topk CLAP descriptors}, single hit, dry
```

Example: `Kick 808 Overdrive.wav` → `kick drum, kick overdrive, distorted,
analog, sub bass, synthetic, single hit, dry`.

## The mapping (output)

- **`captions.csv`** — `filepath,prompt`; consumed by `stable/train.py`.
- **`captions.txt`** — human-readable `name → caption` for all 552 kicks.

## Implementation notes / compatibility patches

The pinned `laion_clap` predates the repo's torch/transformers, so `ClapTagger`
applies three fixes on load:

1. **`torch.load(weights_only=False)`** — the checkpoint carries numpy globals;
   torch ≥ 2.6 defaults to `weights_only=True` and would reject it.
2. **`load_state_dict(strict=False)`** — transformers ≥ 5 dropped RoBERTa's
   `position_ids` buffer that the checkpoint still contains.
3. **tensor input** — `get_audio_embedding_from_data` needs a torch tensor when
   `use_tensor=True`.

**Decoding:** audio is decoded to mono 48 kHz via **ffmpeg**. Note: Ableton
factory `.aif` use a proprietary `able` codec that no open decoder reads — those
must be exported to **WAV** first.
