# ♻️ ScrapNet – Waste Classification System

> Deep learning pipeline that classifies waste images into 7 categories using **Transfer Learning** (EfficientNet-B0/B3) and a **Custom CNN built from scratch** — deployed as a live Streamlit web application.



## 📌 Overview

ScrapNet is a PBL (Project Based Learning) submission that investigates and compares four model architectures for waste image classification:

| Model | Type | Input Size |
|---|---|---|
| EfficientNet-B0 (no augmentation) | Transfer Learning | 224 × 224 |
| EfficientNet-B0 (with augmentation) | Transfer Learning | 224 × 224 |
| EfficientNet-B3 (with augmentation) | Transfer Learning | 300 × 300 |
| **ScrapNetCNN** (from scratch) | Custom CNN | 224 × 224 |

**Waste categories:** Cardboard · Glass · Metal · Paper · Plastic · Trash · Compost

---

## 🗂️ Repository Structure

```
Scrapnet/
├── app.py                  # Streamlit web app (5 pages)
├── predict.py              # Inference module
├── custom_cnn.py           # ScrapNetCNN architecture
├── scrapnet_utils.py       # Shared training utilities
├── requirements.txt        # Python dependencies
│
├── scrapnet_colab.ipynb    # EfficientNet training (Colab)
├── scrapnet_cnn.ipynb      # Custom CNN + 4-model comparison
│
├── artifacts/              # Production model weights (for deployment)
│   ├── efficientnet_b0_waste.pth
│   ├── classes.json
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix_normalized.png
│
├── assets/                 # Plots for GitHub Pages presentation
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix_normalized.png
│   ├── f1_scores.png
│   ├── all_models_comparison.png
│   ├── all_models_per_class_f1.png
│   └── all_models_val_curves.png
│
└── index.html              # GitHub Pages presentation
```

> ⚠️ `experiments/` and `master/` (dataset) are NOT pushed to GitHub — they are too large.
> The `artifacts/` folder contains the production-ready model for deployment.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Bhavya-7-11/Scrapnet.git
cd Scrapnet

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🏋️ Train Models

**Option A — Google Colab (recommended, free GPU):**
1. Upload `scrapnet_utils.py`, `custom_cnn.py`, and your dataset to `MyDrive/ScrapNet/`
2. Open `scrapnet_colab.ipynb` in Colab → Runtime → T4 GPU → Run All
3. Open `scrapnet_cnn.ipynb` → Run All

**Option B — Local Jupyter:**
```bash
pip install jupyter
jupyter notebook scrapnet_local.ipynb
# Edit DATA_DIR to point to your dataset folder, then Run All
```

---

## 🧠 ScrapNetCNN Architecture

```
Input  3 × 224 × 224
  │
  ├─ Block 1 ── Conv(3→32)    BatchNorm  ReLU  MaxPool  →  32 × 112 × 112
  ├─ Block 2 ── Conv(32→64)   BatchNorm  ReLU  MaxPool  →  64 × 56  × 56
  ├─ Block 3 ── Conv(64→128)  BatchNorm  ReLU  MaxPool  → 128 × 28  × 28
  ├─ Block 4 ── Conv(128→256) BatchNorm  ReLU  MaxPool  → 256 × 14  × 14
  │
  ├─ Global Average Pooling   →  256-dim vector
  ├─ FC(256→512)  ReLU  Dropout(0.4)
  ├─ FC(512→256)  ReLU  Dropout(0.3)
  └─ FC(256→num_classes)
```

---

## 📊 Results Summary

| Model | Best Val Acc | Mean F1 |
|---|---|---|
| EfficientNet-B0 (no aug) | — | — |
| EfficientNet-B0 (aug) | ~82% | — |
| EfficientNet-B3 (aug) ★ | ~85% | Best |
| ScrapNetCNN (from scratch) | ~48% | ~0.49 |

*Fill in exact numbers from `experiments/*/summary.json` after training.*

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub (see `.gitignore` section below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → `main` branch → `app.py`
5. Click **Deploy**

The app uses `artifacts/` for the deployed model — make sure those files are pushed.

---

## 📁 .gitignore

The following are excluded from the repo:

```
experiments/        # large training outputs (~GB)
master/             # dataset images (~GB)
venv/
__pycache__/
*.pyc
.DS_Store
```

---

## 🛠️ Tech Stack

- **PyTorch** — model training and inference
- **TorchVision** — EfficientNet pretrained models
- **Streamlit** — web application
- **scikit-learn** — metrics and stratified splitting
- **Matplotlib** — plot generation
- **Pillow** — image processing
- **Google Colab** — GPU training environment

---

## 👤 Author

**Bhavya Bansal**  
Roll: 23FE10CSE00078  
B.Tech CSE · Manipal University Jaipur  
Project Guide: Mahesh Jangid  
Jan–May 2026

---

## 📄 License

MIT License — free to use and modify with attribution.
