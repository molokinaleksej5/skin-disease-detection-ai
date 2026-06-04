from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    seed: int = 42
    deterministic: bool = False
    eval_batch_size: int = 64
    model_variant: str = "resnet50"
    head_type: str = "mlp"  # "linear" или "mlp"
    head_hidden: int = 512
    head_dropout: float = 0.3
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0  # можно потом поставить 0.05

    # config.py находится в .../MedicalNN/src/config.py
    project_dir: Path = Path(__file__).resolve().parents[1]  # => .../MedicalNN
    data_dir: Path = project_dir / "data"

    img_dir: Path = data_dir / "ISIC_2019_Training_Input"
    gt_csv: Path = data_dir / "ISIC_2019_Training_GroundTruth.csv"

    splits_dir: Path = data_dir / "splits"
    train_csv: Path = splits_dir / "train.csv"
    val_csv: Path = splits_dir / "val.csv"
    test_csv: Path = splits_dir / "test.csv"

    models_dir: Path = project_dir / "models"
    best_model_path: Path = models_dir / "best_model.pt"
    label_map_path: Path = models_dir / "label_map.json"

    reports_dir: Path = project_dir / "reports"
    figures_dir: Path = reports_dir / "figures"

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    early_stop_patience: int = 3

CFG = Config()
