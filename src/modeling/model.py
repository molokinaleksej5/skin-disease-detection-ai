import torch
import torch.nn as nn
from torchvision import models

def build_resnet(
    num_classes: int,
    pretrained: bool = True,
    variant: str = "resnet18",
    head_type: str = "linear",
    head_hidden: int = 512,
    head_dropout: float = 0.3,
) -> nn.Module:
    variant = variant.lower()
    head_type = head_type.lower()

    if variant == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif variant == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif variant == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    in_features = model.fc.in_features

    if head_type == "linear":
        model.fc = nn.Linear(in_features, num_classes)
    elif head_type == "mlp":
        model.fc = nn.Sequential(
            nn.Linear(in_features, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")

    return model


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(x)
    return torch.softmax(logits, dim=1)
