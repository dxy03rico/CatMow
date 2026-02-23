# CatMow 🐱

**Real-time cat sound recognition on Raspberry Pi 5**

A 4-class cat sound classifier (meow / purr / yowl / other) that runs on-device using TFLite, trained with MobileNetV2 transfer learning on Mel-spectrogram images.

---

## Classes

| Label | Description |
|-------|-------------|
| `meow` | Standard meow calls (including hungry / happy variants) |
| `purr` | Continuous low-frequency purring |
| `yowl` | Intense vocalizations (yowl + high-quality growl samples) |
| `other` | Background noise / silence |

---

## Architecture

- **Base model**: MobileNetV2 (ImageNet pre-trained, frozen in Phase 1)
- **Input**: 128×128×3 Mel-spectrogram images (1-second audio clips)
- **Output**: 4-class softmax
- **Two-phase training**: Phase 1 — train head (10 epochs, lr=1e-3); Phase 2 — fine-tune top 30 layers (40 epochs, lr=1e-4)
- **Runtime**: TFLite (`cat_sound.tflite`, ~9.8 MB)

---

## Pipeline

```
data/raw/          →  scripts/02_preprocess.py
                   →  data/features/ (X.npy, y.npy, dataset.csv)
                   →  scripts/03_train.py
                   →  models/best_model.keras
                   →  scripts/04_convert_tflite.py
                   →  models/cat_sound.tflite
                   →  pi5/pi5_inference.py  (on Raspberry Pi 5)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare audio data

Place `.mp3` / `.wav` files in:

```
data/raw/
  meow/
  purr/
  yowl/
  background/
  growl/        ← optional, merged into yowl
```

### 3. Filter growl data (optional)

```bash
python scripts/filter_growl.py --dry-run   # preview
python scripts/filter_growl.py --keep 15   # archive low-quality growl
```

### 4. Preprocess

```bash
python scripts/02_preprocess.py
```

### 5. Train

```bash
python scripts/03_train.py
```

### 6. Convert to TFLite

```bash
python scripts/04_convert_tflite.py
```

### 7. Deploy to Pi 5

```bash
scp models/cat_sound.tflite pi5/pi5_inference.py user@<pi-ip>:~/catmow/
```

---

## Raspberry Pi 5 Inference

```bash
python pi5_inference.py
```

Features:
- 3-level VAD pre-filter (RMS energy / cat-band ratio / spectral flatness)
- Per-class confidence thresholds (meow=0.60, purr=0.55, yowl=0.88, other=0.50)
- Event-based debounce with sustain + cooldown
- Emoji output + confidence display

---

## Data

`data/processed/` contains segmented and augmented audio clips (`.mp3`):

| Directory | Contents |
|-----------|----------|
| `growl/` | 15 high-quality growl clips (spectral-filtered) |
| `growl_excluded/` | 16 archived low-quality growl clips |
| `hiss/` | 31 hiss clips (not used in training — spectral overlap with purr) |
| `yowl/` | 11 long-form yowl recordings |
| `purr/` | ~62 purr recordings |
| `meow/` | Meow + hungry + happy clips |
| `background/` | Background noise clips |

---

## Results (v2 model)

| Class | Test Accuracy |
|-------|--------------|
| meow  | 91.3% |
| purr  | 46.0% |
| yowl  | 82.6% |
| other | 2.5% |
| **Overall** | **75.0%** |

*Val accuracy: 86.5% — purr test set is small (74 samples from original distribution) and not representative of real-world diversity.*

---

## Known Issues / Future Work

- **Purr**: training data mostly from YouTube videos → limited diversity. Recording from real cats in different conditions would significantly improve recall.
- **other class**: near-zero test accuracy; needs more varied background noise types.
- **Meow/Yowl boundary**: some confusion remains, especially with tonal meow variants.

---

## License

MIT
