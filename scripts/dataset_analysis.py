import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from src.config import CFG


def main():

    reports_dir = CFG.reports_dir / "dataset_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    gt = pd.read_csv(CFG.gt_csv)

    print("Loading splits...")
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # ----------------------------
    # 1️⃣ Class distribution
    # ----------------------------
    print("Plotting class distribution...")

    class_counts = train_df["label"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution (Train)")
    plt.xlabel("Class index")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=300)
    plt.close()

    # ----------------------------
    # 2️⃣ Correlation matrix
    # ----------------------------
    print("Building correlation matrix...")

    # Берём one-hot из оригинального ground truth
    gt_onehot = gt.drop(columns=["image"])

    corr_matrix = gt_onehot.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Classes")
    plt.tight_layout()
    plt.savefig(reports_dir / "correlation_matrix.png", dpi=300)
    plt.close()

    # ----------------------------
    # 3️⃣ Image size statistics
    # ----------------------------
    print("Calculating image size statistics...")

    widths = []
    heights = []

    for fname in tqdm(train_df["filename"].head(1000)):
        img_path = CFG.img_dir / fname
        if img_path.exists():
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    print("Mean width:", widths.mean())
    print("Mean height:", heights.mean())

    plt.figure(figsize=(6, 5))
    plt.hist(widths, bins=30, alpha=0.7, label="Width")
    plt.hist(heights, bins=30, alpha=0.7, label="Height")
    plt.legend()
    plt.title("Image Size Distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "image_size_distribution.png", dpi=300)
    plt.close()

    # ----------------------------
    # 4️⃣ Pixel statistics
    # ----------------------------
    print("Calculating pixel mean/std...")

    means = []
    stds = []

    for fname in tqdm(train_df["filename"].head(500)):
        img_path = CFG.img_dir / fname
        if img_path.exists():
            img = np.array(Image.open(img_path)) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))

    means = np.array(means)
    stds = np.array(stds)

    print("Dataset mean:", means.mean(axis=0))
    print("Dataset std:", stds.mean(axis=0))

    print("Dataset analysis completed.")
    print("Results saved to:", reports_dir)


if __name__ == "__main__":
    main()
