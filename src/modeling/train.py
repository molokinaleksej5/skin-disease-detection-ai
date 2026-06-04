# src/modeling/train.py
# ✅ Исправлено: корректное сохранение истории в CSV (по каждой архитектуре отдельно),
# ✅ flush/fsync чтобы файл точно обновлялся,
# ✅ сохранение ckpt с head_type/head_hidden/head_dropout/img_size/seed/lr/best_val_acc/epoch,
# ✅ убраны дублирующиеся расчёты class weights (оставлен один),
# ✅ alpha считается, но по умолчанию для focal используется alpha=None (как у тебя лучший результат),
# ✅ меньше шансов зависнуть из-за Excel (файл всегда закрывается после записи).

import os
import csv
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import CFG
from src.data.dataset import ISICDataset
from src.modeling.model import build_resnet
from src.utils.seed import set_seed, seed_worker
from src.utils.losses import FocalLoss


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def _ensure_history_file(history_path):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "run_time", "seed", "variant", "epoch",
                "train_loss", "train_acc", "val_loss", "val_acc",
                "lr", "best_val_acc"
            ])
        f.flush()
        os.fsync(f.fileno())


def _append_history_row(history_path, row):
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _compute_class_weights(train_csv, num_classes: int):
    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        raise ValueError(f"train_csv must have column 'label'. Got: {list(df.columns)}")

    class_counts = (
        df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )

    counts = class_counts.values.astype("float32")
    counts[counts == 0] = 1.0

    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)

    return class_counts.to_dict(), weights


def main() -> None:
    print("=== TRAIN START ===", flush=True)

    # dirs
    CFG.models_dir.mkdir(parents=True, exist_ok=True)
    CFG.reports_dir.mkdir(parents=True, exist_ok=True)

    # seed + generator
    g = set_seed(CFG.seed, deterministic=getattr(CFG, "deterministic", False))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")
    print("Device:", device, flush=True)

    # label_map -> num_classes
    print("Loading label_map...", flush=True)
    with open(CFG.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print("num_classes:", num_classes, flush=True)

    # history path (отдельно для каждой архитектуры)
    variant = CFG.model_variant
    history_path = CFG.reports_dir / f"train_history_{variant}.csv"
    _ensure_history_file(history_path)
    print("History CSV:", history_path.resolve(), flush=True)

    # transforms / data
    print("Preparing transforms...", flush=True)
    train_tf, val_tf = get_transforms(CFG.img_size)

    print("Creating datasets...", flush=True)
    train_ds = ISICDataset(CFG.train_csv, CFG.img_dir, transform=train_tf)
    val_ds = ISICDataset(CFG.val_csv, CFG.img_dir, transform=val_tf)

    print("Creating dataloaders...", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # model
    print("Building model...", flush=True)
    model = build_resnet(
        num_classes=num_classes,
        pretrained=True,
        variant=variant,
        head_type=CFG.head_type,
        head_hidden=CFG.head_hidden,
        head_dropout=CFG.head_dropout,
    ).to(device)
    print("Model built successfully.", flush=True)

    # class weights (один раз)
    print("Computing class weights from:", str(CFG.train_csv), flush=True)
    class_counts_dict, class_weights = _compute_class_weights(CFG.train_csv, num_classes)
    print("Class counts:", class_counts_dict, flush=True)
    print("Class weights (normalized):", class_weights, flush=True)
    alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # loss
    use_focal = bool(getattr(CFG, "use_focal_loss", True))
    gamma = float(getattr(CFG, "focal_gamma", 2.0))
    ls = float(getattr(CFG, "label_smoothing", 0.0))

    if use_focal:
        # ВАЖНО: по твоим лучшим результатам alpha=None (без дополнительного перекоса)
        criterion = FocalLoss(alpha=None, gamma=gamma, label_smoothing=ls)
        print(f"Using FocalLoss(alpha=None, gamma={gamma}, label_smoothing={ls})", flush=True)
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=ls)
        print(f"Using CrossEntropyLoss(weight=class_weights, label_smoothing={ls})", flush=True)

    # optimizer/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_acc = -1.0
    patience_left = int(getattr(CFG, "early_stop_patience", 10**9))  # можно ставить большое, если хочешь без early stop

    print("Starting training loop...", flush=True)

    for epoch in range(1, int(CFG.epochs) + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.epochs} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = int(y.size(0))
            train_loss_sum += float(loss.item()) * bs
            preds = logits.detach().argmax(dim=1)
            train_correct += int((preds == y).sum().item())
            train_total += bs

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{CFG.epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)

                bs = int(y.size(0))
                val_loss_sum += float(loss.item()) * bs
                preds = logits.argmax(dim=1)
                val_correct += int((preds == y).sum().item())
                val_total += bs

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        scheduler.step(val_acc)

        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6g}",
            flush=True
        )

        # best / early stop
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_left = int(getattr(CFG, "early_stop_patience", 10**9))

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "variant": variant,
                    "head_type": CFG.head_type,
                    "head_hidden": int(CFG.head_hidden),
                    "head_dropout": float(CFG.head_dropout),
                    "img_size": int(CFG.img_size),
                    "seed": int(CFG.seed),
                    "epoch": int(epoch),
                    "lr": float(current_lr),
                    "best_val_acc": float(best_val_acc),
                    "use_focal_loss": bool(use_focal),
                    "focal_gamma": float(gamma),
                    "label_smoothing": float(ls),
                },
                CFG.best_model_path,
            )
            print("Saved best model to:", str(CFG.best_model_path), flush=True)
        else:
            patience_left -= 1

        # write history AFTER best_val_acc updated
        _append_history_row(
            history_path,
            [
                datetime.now().isoformat(timespec="seconds"),
                int(CFG.seed),
                str(variant),
                int(epoch),
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{current_lr:.8f}",
                f"{best_val_acc:.6f}",
            ],
        )

        if patience_left <= 0:
            print("Early stopping.", flush=True)
            break

    print("=== TRAIN END ===", flush=True)


if __name__ == "__main__":
    main()
