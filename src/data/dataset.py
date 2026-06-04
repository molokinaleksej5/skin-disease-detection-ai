from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

        if "filename" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV должен содержать столбцы: filename, label")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["filename"]
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
