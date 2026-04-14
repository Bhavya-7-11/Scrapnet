"""
scrapnet_utils.py
=================
Shared utilities for ScrapNet – Waste Classification System.
Contains: data loading, model building, training loop, evaluation, and plotting.
All functions are self-contained and reusable across experiments.
"""

import os
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ──────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# 2. Device detection
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-select CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


# ──────────────────────────────────────────────
# 3. Transforms
# ──────────────────────────────────────────────

def get_transforms(img_size: int, augment: bool):
    """
    Returns (train_tfms, val_tfms).
    If augment=False, train_tfms == val_tfms (baseline).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        train_tfms = val_tfms  # no augmentation for baseline

    return train_tfms, val_tfms


# ──────────────────────────────────────────────
# 4. Dataset / DataLoaders
# ──────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    train_tfms,
    val_tfms,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
):
    """
    Stratified 70/15/15 train-val-test split.
    Returns: train_loader, val_loader, test_loader, classes, class_weights (tensor on CPU)
    """
    # Base dataset (no transform) – used only for labels
    base_ds = datasets.ImageFolder(data_dir)
    classes = base_ds.classes
    y = np.array([label for _, label in base_ds.samples])
    idx = np.arange(len(base_ds))

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_split + test_split, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, y))

    # Second split: val vs test (equal halves of the remainder)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, y[temp_idx]))
    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    print(f"[Data] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"[Data] Classes: {classes}")

    # Build subsets with correct transforms
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_tfms), train_idx)
    val_ds   = Subset(datasets.ImageFolder(data_dir, transform=val_tfms),   val_idx)
    test_ds  = Subset(datasets.ImageFolder(data_dir, transform=val_tfms),   test_idx)

    # Compute class weights (inverse frequency) from training labels only
    train_labels = y[train_idx]
    counts = Counter(train_labels)
    num_classes = len(classes)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float32,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes, class_weights


# ──────────────────────────────────────────────
# 5. Model Building
# ──────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Build a pretrained EfficientNet (b0 or b3) with a custom classification head.
    model_name: 'b0' or 'b3'
    """
    if model_name == "b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif model_name == "b3":
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'b0' or 'b3'.")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] EfficientNet-{model_name.upper()} | Params: {total_params:,}")
    return model


# ──────────────────────────────────────────────
# 6. Training & Evaluation
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for x, yb in loader:
        x, yb = x.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc  += (logits.argmax(1) == yb).float().mean().item()
    return running_loss / len(loader), running_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    for x, yb in loader:
        x, yb = x.to(device), yb.to(device)
        logits = model(x)
        running_loss += criterion(logits, yb).item()
        running_acc  += (logits.argmax(1) == yb).float().mean().item()
    return running_loss / len(loader), running_acc / len(loader)


@torch.no_grad()
def predict_all(model, loader, device):
    """Returns (y_true, y_pred) as numpy arrays over the full loader."""
    model.eval()
    all_true, all_pred = [], []
    for x, yb in loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
    return np.array(all_true), np.array(all_pred)


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs: int = 15,
    lr: float = 1e-4,
    patience: int = 4,
):
    """
    Full training loop with early stopping and LR scheduling.
    Returns: (trained model with best weights, history dict)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state   = None
    bad_epochs   = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f}  acc {va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs   = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"  [Early Stop] Triggered at epoch {epoch}. Best val acc: {best_val_acc:.4f}")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.to(device)
    print(f"  [Done] Best val acc: {best_val_acc:.4f}")
    return model, history


# ──────────────────────────────────────────────
# 7. Plotting & Saving Results
# ──────────────────────────────────────────────

def plot_loss_curve(history: dict, save_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "s--", label="Val Loss",   linewidth=2)
    ax.set_title("Loss vs Epochs", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=300)
    plt.close(fig)


def plot_accuracy_curve(history: dict, save_dir: Path) -> None:
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], "o-", label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"],   "s--", label="Val Acc",   linewidth=2)
    ax.set_title("Accuracy vs Epochs", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_curve.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_dir: Path, normalized: bool = False) -> None:
    data   = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalized else cm
    fmt    = ".2f" if normalized else "d"
    title  = "Confusion Matrix (Normalized)" if normalized else "Confusion Matrix (Counts)"
    fname  = "confusion_matrix_normalized.png" if normalized else "confusion_matrix.png"
    thresh = data.max() / 2.0

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=14)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val  = data[i, j]
            text = f"{val:{fmt}}" if normalized else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_dir / fname, dpi=300)
    plt.close(fig)


def plot_f1_scores(y_true, y_pred, classes: list, save_dir: Path) -> None:
    _, _, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    colors = ["#e74c3c" if v < 0.7 else "#f39c12" if v < 0.85 else "#2ecc71" for v in f1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, f1, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Per-Class F1 Score", fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(np.mean(f1), color="navy", linestyle="--", linewidth=1.5, label=f"Mean F1 = {np.mean(f1):.3f}")
    ax.legend()
    ax.set_xticklabels(classes, rotation=45, ha="right")
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "f1_scores.png", dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────
# 8. Full Experiment Runner
# ──────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    model_name: str,
    img_size: int,
    augment: bool,
    data_dir: str,
    base_output_dir: str,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-4,
    patience: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    device: torch.device = None,
):
    """
    End-to-end experiment runner.
    Trains a model and saves all outputs to base_output_dir/exp_name/.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Model: EfficientNet-{model_name.upper()} | Augment: {augment} | IMG: {img_size}x{img_size}")
    print(f"{'='*60}")

    if device is None:
        device = get_device()

    seed_everything(seed)

    # Output directory
    save_dir = Path(base_output_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_tfms, val_tfms = get_transforms(img_size, augment)

    # Data
    train_loader, val_loader, test_loader, classes, class_weights = build_dataloaders(
        data_dir=data_dir,
        train_tfms=train_tfms,
        val_tfms=val_tfms,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_classes = len(classes)

    # Model + loss
    model     = build_model(model_name, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Train
    model, history = run_training(
        model, train_loader, val_loader, criterion, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    # Save model + metadata
    torch.save(model.state_dict(), save_dir / "model.pth")
    with open(save_dir / "classes.json", "w") as f:
        json.dump(classes, f)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [Saved] model.pth + classes.json + history.json → {save_dir}")

    # Plots
    plot_loss_curve(history, save_dir)
    plot_accuracy_curve(history, save_dir)

    # Evaluate on test set
    y_true, y_pred = predict_all(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, classes, save_dir, normalized=False)
    plot_confusion_matrix(cm, classes, save_dir, normalized=True)
    plot_f1_scores(y_true, y_pred, classes, save_dir)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\n[Classification Report]\n", report)
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Summary metrics
    _, _, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    summary = {
        "experiment":    exp_name,
        "model":         model_name,
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
