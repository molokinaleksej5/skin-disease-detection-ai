# scripts/07_plot_histories.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"
OUT_DIR = REPORTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILES = {
    "resnet18": REPORTS_DIR / "train_history_resnet18.csv",
    "resnet34": REPORTS_DIR / "train_history_resnet34.csv",
    "resnet50": REPORTS_DIR / "train_history_resnet50.csv",
}

REQ_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"History not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}. Got columns: {list(df.columns)}"
        )

    for c in REQ_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def plot_one(histories: dict, metric: str, title: str, ylabel: str, out_path: Path):
    plt.figure()
    for name, df in histories.items():
        plt.plot(df["epoch"], df[metric], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("Saved:", out_path)


def main():
    histories = {}
    for name, path in HISTORY_FILES.items():
        df = load_history(path)
        histories[name] = df
        print(f"{name}: epochs={df['epoch'].max()} file={path}")

    # ✅ РОВНО 4 ГРАФИКА
    plot_one(histories, "train_loss", "Train loss (ResNet18/34/50)", "Train loss",
             OUT_DIR / "train_loss_all.png")

    plot_one(histories, "val_loss", "Validation loss (ResNet18/34/50)", "Validation loss",
             OUT_DIR / "val_loss_all.png")

    plot_one(histories, "train_acc", "Train accuracy (ResNet18/34/50)", "Train accuracy",
             OUT_DIR / "train_acc_all.png")

    plot_one(histories, "val_acc", "Validation accuracy (ResNet18/34/50)", "Validation accuracy",
             OUT_DIR / "val_acc_all.png")

    print("Done. Output:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
