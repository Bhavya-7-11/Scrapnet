"""
custom_cnn.py
=============
ScrapNet – Custom CNN built entirely from scratch using PyTorch.

This file contains:
  1. ScrapNetCNN  – the model class
  2. run_cnn_experiment – end-to-end trainer that mirrors run_experiment()
                          in scrapnet_utils.py so results are directly comparable.

HOW THE ARCHITECTURE WORKS
───────────────────────────
Input image  →  3 × 224 × 224

Block 1 (shallow features)
  Conv2d(3→32, 3×3)  →  BatchNorm  →  ReLU  →  MaxPool(2×2)
  Output: 32 × 112 × 112

Block 2 (mid-level features)
  Conv2d(32→64, 3×3) →  BatchNorm  →  ReLU  →  MaxPool(2×2)
  Output: 64 × 56 × 56

Block 3 (deeper features)
  Conv2d(64→128, 3×3) → BatchNorm  →  ReLU  →  MaxPool(2×2)
  Output: 128 × 28 × 28

Block 4 (high-level features)
  Conv2d(128→256, 3×3) → BatchNorm → ReLU  →  MaxPool(2×2)
  Output: 256 × 14 × 14

Global Average Pooling  →  256-dim vector   (replaces Flatten+large FC)

Classifier head
  FC(256→512)  →  ReLU  →  Dropout(0.4)
  FC(512→256)  →  ReLU  →  Dropout(0.3)
  FC(256→num_classes)   ← raw logits (CrossEntropyLoss applies Softmax internally)

WHY THESE DESIGN CHOICES?
──────────────────────────
• BatchNorm after every conv  – normalises activations so training is stable
                                and faster; acts as a mild regulariser.
• ReLU                        – simple, fast, avoids vanishing gradients.
• MaxPool(2×2)                – halves spatial dims, keeps the strongest signal.
• 4 blocks instead of 3       – extra depth helps capture texture/shape cues
                                needed for waste classification.
• AdaptiveAvgPool at the end  – model works for any input size (224, 300, …).
• Dropout(0.4 / 0.3)          – randomly zeroes activations during training to
                                prevent the FC layers from memorising the data.
• Raw logits output            – PyTorch's CrossEntropyLoss already applies
                                log-softmax internally; double-softmax would hurt.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Re-use all the shared utilities from scrapnet_utils
from scrapnet_utils import (
    get_device,
    seed_everything,
    get_transforms,
    build_dataloaders,
    run_training,
    predict_all,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_f1_scores,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ScrapNetCNN(nn.Module):
    """
    Custom CNN for waste image classification.

    Parameters
    ----------
    num_classes : int
        Number of output categories (e.g. 7 for cardboard/glass/metal/…).
    dropout1 : float
        Dropout rate after the first FC layer (default 0.4).
    dropout2 : float
        Dropout rate after the second FC layer (default 0.3).
    """

    def __init__(self, num_classes: int, dropout1: float = 0.4, dropout2: float = 0.3):
        super().__init__()

        # ── Convolutional feature extractor ───────────────────────────────────
        # Each block: Conv → BatchNorm → ReLU → MaxPool
        # Channels double every block: 32 → 64 → 128 → 256

        self.features = nn.Sequential(

            # ── Block 1 – detect edges & simple textures ──────────────────────
            nn.Conv2d(in_channels=3,  out_channels=32,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # normalise across the batch for each channel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224→112

            # ── Block 2 – detect shapes & colour regions ──────────────────────
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112→56

            # ── Block 3 – detect object parts ─────────────────────────────────
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56→28

            # ── Block 4 – detect complex patterns ─────────────────────────────
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28→14
        )

        # ── Global Average Pooling ─────────────────────────────────────────────
        # Collapses the 14×14 spatial grid into a single number per channel.
        # Output: (batch, 256)
        # Why? Much fewer parameters than Flatten→FC; also regularises the model.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classifier head ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),   # randomly zero 40% of neurons while training

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),   # randomly zero 30% of neurons while training

            nn.Linear(256, num_classes),  # raw logits – no softmax here!
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape : (batch, 3, H, W)
        returns  : (batch, num_classes)  ← raw logits
        """
        x = self.features(x)            # (B, 256, 14, 14)
        x = self.global_avg_pool(x)     # (B, 256, 1, 1)
        x = x.flatten(start_dim=1)      # (B, 256)
        x = self.classifier(x)          # (B, num_classes)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.  EXPERIMENT RUNNER  (mirrors run_experiment from scrapnet_utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_cnn_experiment(
    exp_name: str,
    img_size: int,
    augment: bool,
    data_dir: str,
    base_output_dir: str,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,        # CNNs trained from scratch usually need a higher LR
    patience: int = 5,
    num_workers: int = 2,
    seed: int = 42,
    device: torch.device = None,
    dropout1: float = 0.4,
    dropout2: float = 0.3,
):
    """
    Train ScrapNetCNN from scratch and save all outputs.

    Saves to  base_output_dir / exp_name /
      model.pth, classes.json, history.json, summary.json,
      loss_curve.png, accuracy_curve.png,
      confusion_matrix.png, confusion_matrix_normalized.png,
      f1_scores.png, classification_report.txt

    Returns
    -------
    model   : trained ScrapNetCNN
    history : dict with train_loss / train_acc / val_loss / val_acc
    summary : dict with key metrics
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Model: Custom ScrapNetCNN | Augment: {augment} | IMG: {img_size}x{img_size}")
    print(f"{'='*60}")

    if device is None:
        device = get_device()

    seed_everything(seed)

    # ── Output directory ──────────────────────────────────────────────────────
    save_dir = Path(base_output_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tfms, val_tfms = get_transforms(img_size, augment)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, classes, class_weights = build_dataloaders(
        data_dir=data_dir,
        train_tfms=train_tfms,
        val_tfms=val_tfms,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_classes = len(classes)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ScrapNetCNN(num_classes=num_classes, dropout1=dropout1, dropout2=dropout2)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] ScrapNetCNN | Params: {total_params:,}")

    # ── Loss  (weighted so rare classes aren't ignored) ───────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Train ─────────────────────────────────────────────────────────────────
    model, history = run_training(
        model, train_loader, val_loader, criterion, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    # ── Save weights + metadata ───────────────────────────────────────────────
    torch.save(model.state_dict(), save_dir / "model.pth")
    with open(save_dir / "classes.json", "w") as f:
        json.dump(classes, f)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [Saved] model.pth + classes.json + history.json → {save_dir}")

    # ── Plots: loss & accuracy curves ─────────────────────────────────────────
    plot_loss_curve(history, save_dir)
    plot_accuracy_curve(history, save_dir)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_true, y_pred = predict_all(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, classes, save_dir, normalized=False)
    plot_confusion_matrix(cm, classes, save_dir, normalized=True)
    plot_f1_scores(y_true, y_pred, classes, save_dir)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\n[Classification Report]\n", report)
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    _, _, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    summary = {
        "experiment":    exp_name,
        "model":         "custom_cnn",
        "augment":       augment,
        "img_size":      img_size,
        "best_val_acc":  round(max(history["val_acc"]), 4),
        "final_val_acc": round(history["val_acc"][-1], 4),
        "mean_f1":       round(float(np.mean(f1_scores)), 4),
        "per_class_f1":  {cls: round(float(v), 4) for cls, v in zip(classes, f1_scores)},
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [Summary] Best val acc: {summary['best_val_acc']} | Mean F1: {summary['mean_f1']}")
    print(f"  [All outputs saved to: {save_dir}]\n")

    return model, history, summary
