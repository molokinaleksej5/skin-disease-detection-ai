import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import CFG
from src.data.dataset import ISICDataset
from src.modeling.model import build_resnet
from src.utils.seed import set_seed, seed_worker


def load_label_map():
    with open(CFG.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # {"AK":0, ...}
    inv = {int(v): k for k, v in label_map.items()}  # {0:"AK", ...}
    class_names = [inv[i] for i in range(len(inv))]
    return label_map, inv, class_names


def get_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_confidence_hist(conf: np.ndarray, save_path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(conf, bins=30)
    plt.title("Histogram of top-1 confidence (Test)")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_roc_mel(y_true: np.ndarray, y_proba: np.ndarray, mel_index: int, save_path: Path):
    # ROC для MEL vs остальные
    y_true_bin = (y_true == mel_index).astype(int)
    y_score = y_proba[:, mel_index]

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve (MEL vs rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def plot_pr_mel(y_true: np.ndarray, y_proba: np.ndarray, mel_index: int, save_path: Path):
    # y_true: истинные метки (0..K-1)
    # y_proba: вероятности softmax размера [N, K]
    y_true_bin = (y_true == mel_index).astype(int)
    y_score = y_proba[:, mel_index]

    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.title("Precision–Recall curve (MEL vs rest)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()



def main():

    CFG.reports_dir.mkdir(parents=True, exist_ok=True)
    CFG.figures_dir.mkdir(parents=True, exist_ok=True)
    g = set_seed(CFG.seed, deterministic=CFG.deterministic)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = (device.type == "cuda")
    num_workers = CFG.num_workers if device.type == "cuda" else 0

    print("Device:", device)

    label_map, inv, class_names = load_label_map()
    num_classes = len(class_names)


    ckpt = torch.load(CFG.best_model_path, map_location=device)
    variant = ckpt["variant"]
    head_type = ckpt["head_type"]
    head_hidden = int(ckpt["head_hidden"])
    head_dropout = float(ckpt["head_dropout"])

    img_size = int(ckpt.get("img_size", CFG.img_size))

    model = build_resnet(
        num_classes=num_classes,
        pretrained=False,
        variant=variant,
        head_type=head_type,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )
    print("CKPT:", {
        "variant": variant,
        "head_type": head_type,
        "head_hidden": head_hidden,
        "head_dropout": head_dropout,
        "img_size": img_size,
    }, flush=True)

    print("State dict keys sample:",
          [k for k in ckpt["model_state"].keys() if k.startswith("fc.")][:10],
          flush=True)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()


    tf = get_eval_transform(img_size)
    test_ds = ISICDataset(CFG.test_csv, CFG.img_dir, transform=tf)

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    y_true = []
    y_pred = []
    y_conf = []
    y_proba_all = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x = x.to(device)
            logits = model(x)
            proba = torch.softmax(logits, dim=1)

            pred = torch.argmax(proba, dim=1).cpu().numpy()
            conf = torch.max(proba, dim=1).values.cpu().numpy()

            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())
            y_conf.extend(conf.tolist())
            y_proba_all.append(proba.cpu().numpy())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_conf = np.array(y_conf, dtype=float)
    y_proba_all = np.concatenate(y_proba_all, axis=0)


    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {acc:.6f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)


    cm_path = CFG.figures_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    print("Saved:", cm_path)

    conf_path = CFG.figures_dir / "confidence_hist.png"
    plot_confidence_hist(y_conf, conf_path)
    print("Saved:", conf_path)


    if "MEL" in label_map:
        mel_index = int(label_map["MEL"])
        roc_path = CFG.figures_dir / "roc_mel.png"
        plot_roc_mel(y_true, y_proba_all, mel_index, roc_path)
        print("Saved:", roc_path)
        pr_path = CFG.figures_dir / "pr_mel.png"
        plot_pr_mel(y_true, y_proba_all, mel_index, pr_path)
        print("Saved:", pr_path)

    else:
        print("ROC(MEL vs rest) skipped: class 'MEL' not found in label_map.")

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
