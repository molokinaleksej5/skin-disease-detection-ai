import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.config import CFG
from src.modeling.model import build_resnet


def predict_image(image_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load checkpoint ===
    ckpt = torch.load(CFG.best_model_path, map_location=device)

    # === Build model exactly as during training ===
    model = build_resnet(
        num_classes=ckpt["num_classes"],
        pretrained=False,
        variant=ckpt["variant"],
        head_type=ckpt.get("head_type", "linear"),
        head_hidden=ckpt.get("head_hidden", 512),
        head_dropout=ckpt.get("head_dropout", 0.0),
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # === Load label map ===
    with open(CFG.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_class = {v: k for k, v in label_map.items()}

    # === Transform ===
    transform = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = idx_to_class[pred_idx]
    pred_prob = probs[pred_idx].item()

    all_probs = {
        idx_to_class[i]: float(probs[i].item())
        for i in range(len(probs))
    }

    return {
        "predicted_class": pred_class,
        "probability": pred_prob,
        "all_probs": all_probs,
    }