"""
predict.py – ScrapNet Inference Module
=======================================
Supports two model sources:
  1. artifacts/efficientnet_b0_waste.pth  (Streamlit Cloud deployment)
  2. experiments/<exp_name>/model.pth     (after running training notebooks)

Usage:
    from predict import load_model, predict_image
    model, classes = load_model()                    # uses artifacts/ by default
    label, conf, probs = predict_image(model, classes, image)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

IMG_SIZE = 224

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(model_path=None, classes_path=None, device="cpu"):
    """
    Load EfficientNet-B0 with trained weights.

    Parameters
    ----------
    model_path   : str or Path, optional
        Path to .pth weights file.
        Defaults to  artifacts/efficientnet_b0_waste.pth
        Falls back to experiments/b0_aug/model.pth  if artifacts/ not found.
    classes_path : str or Path, optional
        Path to classes.json.
        Defaults to  artifacts/classes.json
        Falls back to experiments/b0_aug/classes.json  if artifacts/ not found.
    device       : str  ("cpu" or "cuda")

    Returns
    -------
    model   : nn.Module  (eval mode)
    classes : list[str]
    """
    repo_root = Path(__file__).resolve().parent

    # ── Resolve model path ────────────────────────────────────────────────────
    if model_path is None:
        candidate_artifacts = repo_root / "artifacts" / "efficientnet_b0_waste.pth"
        candidate_exp       = repo_root / "experiments" / "b0_aug" / "model.pth"
        if candidate_artifacts.exists():
            model_path = candidate_artifacts
        elif candidate_exp.exists():
            model_path = candidate_exp
        else:
            raise FileNotFoundError(
                "No model weights found. Expected one of:\n"
                f"  {candidate_artifacts}\n"
                f"  {candidate_exp}\n"
                "Run scrapnet_colab.ipynb or scrapnet_local.ipynb first."
            )
    else:
        model_path = Path(model_path)

    # ── Resolve classes path ──────────────────────────────────────────────────
    if classes_path is None:
        candidate_artifacts = repo_root / "artifacts" / "classes.json"
        candidate_exp       = repo_root / "experiments" / "b0_aug" / "classes.json"
        if candidate_artifacts.exists():
            classes_path = candidate_artifacts
        elif candidate_exp.exists():
            classes_path = candidate_exp
        else:
            raise FileNotFoundError(
                "classes.json not found. Expected one of:\n"
                f"  {candidate_artifacts}\n"
                f"  {candidate_exp}"
            )
    else:
        classes_path = Path(classes_path)

    # ── Load classes ──────────────────────────────────────────────────────────
    with open(classes_path, "r") as f:
        classes = json.load(f)

    # ── Build model ───────────────────────────────────────────────────────────
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model, classes


@torch.no_grad()
def predict_image(model, classes, image: Image.Image, device="cpu"):
    """
    Classify a single PIL image.

    Parameters
    ----------
    model   : nn.Module returned by load_model()
    classes : list[str] returned by load_model()
    image   : PIL.Image.Image
    device  : str

    Returns
    -------
    label : str    predicted class name
    conf  : float  confidence (0–1)
    probs : np.ndarray  full probability vector over all classes
    """
    x = val_tfms(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    import numpy as np
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), probs
