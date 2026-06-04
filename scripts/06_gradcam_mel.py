from pathlib import Path
import json
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from src.config import CFG
from src.modeling.model import build_resnet
from src.interpretability.gradcam import generate_gradcam


@torch.no_grad()
def predict_class(image_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CFG.best_model_path, map_location=device)

    model = build_resnet(
        num_classes=ckpt["num_classes"],
        pretrained=False,
        variant=ckpt["variant"],
        head_type=ckpt["head_type"],
        head_hidden=ckpt["head_hidden"],
        head_dropout=ckpt["head_dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return pred, conf


if __name__ == "__main__":
    label_map = json.load(open(CFG.label_map_path, encoding="utf-8"))
    mel_idx = int(label_map["MEL"])
    print("MEL index:", mel_idx)

    df = pd.read_csv(CFG.test_csv)

    # MEL по истинной метке
    mel_df = df[df["label"] == mel_idx].copy()
    print("MEL in test:", len(mel_df))

    out_dir = CFG.reports_dir / "gradcam_mel"
    out_dir.mkdir(parents=True, exist_ok=True)

    tp = []
    fn = []

    for _, row in mel_df.iterrows():
        filename = row["filename"]          # уже с .jpg
        img_path = CFG.img_dir / filename   # НЕ добавляем .jpg

        if not img_path.exists():
            continue

        pred, conf = predict_class(img_path)

        if pred == mel_idx and len(tp) < 3:
            tp.append((filename, pred, conf))
        elif pred != mel_idx and len(fn) < 3:
            fn.append((filename, pred, conf))

        if len(tp) >= 3 and len(fn) >= 3:
            break

    print("TP:", tp)
    print("FN:", fn)

    # ВАЖНО: target_class = mel_idx (строим CAM именно для MEL)
    for filename, pred, conf in tp:
        img_path = CFG.img_dir / filename
        out_path = out_dir / f"{filename}_TP_pred{pred}_conf{conf:.2f}.jpg"
        generate_gradcam(img_path, out_path, target_class=mel_idx)

    for filename, pred, conf in fn:
        img_path = CFG.img_dir / filename
        out_path = out_dir / f"{filename}_FN_pred{pred}_conf{conf:.2f}.jpg"
        generate_gradcam(img_path, out_path, target_class=mel_idx)

    print("Saved to:", out_dir)
