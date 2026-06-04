import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG

CLASSES_DEFAULT = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def build_label_from_onehot(row: pd.Series, classes: list[str]) -> str:
    # В ISIC_2019_Training_GroundTruth.csv обычно one-hot столбцы по классам
    # и столбец image (id). Берём класс с максимальным значением.
    values = row[classes].astype(float).values
    idx = int(np.argmax(values))
    return classes[idx]

def main() -> None:
    CFG.splits_dir.mkdir(parents=True, exist_ok=True)
    CFG.models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(CFG.seed)

    df = pd.read_csv(CFG.gt_csv)

    # определяем список классов: либо стандартный, либо по csv
    classes = [c for c in CLASSES_DEFAULT if c in df.columns]
    if not classes:
        # если вдруг названия другие — возьмём все колонки кроме image
        classes = [c for c in df.columns if c.lower() != "image"]

    if "image" not in df.columns:
        raise ValueError("В GroundTruth.csv должен быть столбец 'image' (id изображения).")

    df["label_str"] = df.apply(lambda r: build_label_from_onehot(r, classes), axis=1)

    # image_id -> filename
    df["filename"] = df["image"].astype(str) + ".jpg"

    # оставляем только то, что реально есть в папке
    img_set = set(p.name for p in Path(CFG.img_dir).glob("*.jpg"))
    df = df[df["filename"].isin(img_set)].reset_index(drop=True)

    # кодировка меток
    label_map = {name: i for i, name in enumerate(sorted(df["label_str"].unique()))}
    df["label"] = df["label_str"].map(label_map).astype(int)

    # train/val/test (70/15/15)
    train_df, temp_df = train_test_split(
        df[["filename", "label"]],
        test_size=0.30,
        random_state=CFG.seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=CFG.seed,
        stratify=temp_df["label"],
    )

    train_df.to_csv(CFG.train_csv, index=False)
    val_df.to_csv(CFG.val_csv, index=False)
    test_df.to_csv(CFG.test_csv, index=False)

    with open(CFG.label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("OK: splits saved to:", CFG.splits_dir)
    print("OK: label_map saved to:", CFG.label_map_path)
    print("Counts:", len(train_df), len(val_df), len(test_df))
    print("Classes:", label_map)

if __name__ == "__main__":
    main()
