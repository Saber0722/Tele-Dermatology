# 🩺 HAM10000 Multi-Modal Skin Lesion Classification

This project explores multi-class skin lesion classification using the **HAM10000 dataset** with a **multi-modal deep learning architecture** combining:

- 📷 Dermoscopic images
- 🧾 Clinical metadata (age, sex, localization)

The goal is to investigate how different imbalance handling strategies affect melanoma detection performance.

---

# 📁 Project Structure

```

.
│
├── data/
│   ├── raw/                # Original downloaded datasets
│   └── processed/          # Train/Val splits + saved artifacts
│
├── notebooks/
│   ├── data_prep.ipynb
│   ├── ham_eda.ipynb
│   ├── ham_merger.ipynb
│   ├── compare_models.ipynb
│   ├── inference_*.ipynb
│
├── src/
│   ├── train_baseline.py
│   ├── train_baseline_1.5.py
│   ├── training_augmentation.py
│   ├── train_focal.py
│
├── results/
│   └── experiments.csv
│
└── README.md

```

---

# 📊 Dataset

### HAM10000

- 10,015 dermoscopic images
- 7 diagnostic classes:
  - `akiec`
  - `bcc`
  - `bkl`
  - `df`
  - `mel`
  - `nv`
  - `vasc`

### Dataset Statistics

- Total images: **10,015**
- Unique lesions: **7,470**
- Train images: **7,974**
- Validation images: **2,041**
- Lesion overlap: **0** (proper lesion-level split)

Class imbalance ratio: ~58:1

Melanoma percentage: ~11%

---

# 🧠 Model Architecture

## MultiModalNet

Backbone:
- EfficientNet-B0 (ImageNet pretrained)

Metadata Branch:
- MLP (64 → 32)

Fusion:
- Concatenation (Image features + Metadata features)
- FC → ReLU → Dropout → 7-class output

Loss:
- CrossEntropy (baseline)
- Weighted CE
- Focal Loss (experiments)

---

# 🧪 Experiments Conducted

Multiple imbalance strategies have been tested:

1. **Baseline CE**
2. **CE + Increased Melanoma Weight (×1.5)**
3. **CE + Melanoma Weight + Targeted Augmentation**
4. **Focal Loss (with alpha)**
5. **Focal Loss (no alpha)**

All results are stored in:

```

results/experiments.csv

```

---

# 📈 Experiment Results

| Model | Accuracy | Macro F1 | Notes |
|--------|----------|----------|-------|
| Baseline CE | ~0.83 | ~0.79 | Most stable |
| CE + Mel ×1.5 | ~0.84 | ~0.73 | Improved mel recall |
| Augmented | ~0.78 | ~0.69 | Higher mel recall, lower precision |
| Focal (alpha) | ~0.59 | ~0.60 | Overcorrected imbalance |
| Focal (no alpha) | ~0.64 | ~0.61 | Still unstable |

---

## 🔬 Melanoma Recall Evolution

| Model | Mel Recall |
|--------|------------|
| Baseline | ~0.55 |
| CE + Mel ×1.5 | ~0.65 |
| Augmented | ~0.78 |
| Focal | ~0.88 |

Increasing sensitivity caused precision collapse in focal models.

---

# 📊 Model Comparison Visualization

See:

```

notebooks/compare_models.ipynb

````

Includes:

- Overall metric comparison
- Melanoma recall comparison
- Per-class F1 heatmap
- Radar plots

---

# 🧬 Key Findings

1. Simple weighted CrossEntropy performed best overall.
2. Over-aggressive imbalance correction (Focal + weights) destabilized training.
3. Targeted augmentation increased melanoma sensitivity but reduced precision.
4. EfficientNet + moderate class weighting achieved best trade-off.

---

# 🏥 Clinical Perspective

- Baseline CE: Best overall balance
- Weighted CE: Improved melanoma detection
- Focal: High sensitivity but too many false positives

---


# ⚙️ Installation

Using `uv`:

```bash
uv sync
````

---

# 🏃 Training

Baseline:

```bash
python src/train_baseline.py
```

Weighted:

```bash
python src/train_baseline_1.5.py
```

Augmented:

```bash
python src/training_augmentation.py
```

Focal:

```bash
python src/train_focal.py
```

---

# 📌 Reproducibility

All:

* Train/val splits
* Class weights
* Label encoding
* Metrics

Are saved in `data/processed/`.

All experiment metrics appended to:

```
results/experiments.csv
```

---

# 📜 License

Academic research and educational use.

```
