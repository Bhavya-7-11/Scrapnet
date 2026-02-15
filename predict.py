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

def load_model(
    model_path=None,
    classes_path=None,
    device="cpu"
):
    # repo_root = folder where predict.py exists
    repo_root = Path(__file__).resolve().parent

    # default to repo-relative artifact paths
    if model_path is None:
        model_path = repo_root / "artifacts" / "efficientnet_b0_waste.pth"
    else:
        model_path = Path(model_path)

    if classes_path is None:
        classes_path = repo_root / "artifacts" / "classes.json"
    else:
        classes_path = Path(classes_path)

    with open(classes_path, "r") as f:
        classes = json.load(f)

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
    x = val_tfms(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), probs