import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.modeling.model import build_resnet
from src.config import CFG


def generate_gradcam(image_path, output_path, target_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(CFG.best_model_path, map_location=device)

    model = build_resnet(
        num_classes=ckpt["num_classes"],
        pretrained=False,
        variant=ckpt["variant"],
        head_type=ckpt.get("head_type", "linear"),
        head_hidden=int(ckpt.get("head_hidden", 512)),
        head_dropout=float(ckpt.get("head_dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Target layer (последний conv блок ResNet)
    target_layers = [model.layer4[-1]]

    # Transform
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

    # Predict class + confidence
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_class].item())

    # Если target_class не задан — CAM по предсказанию
    if target_class is None:
        target_class = pred_class

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(int(target_class))],
    )[0]

    # Convert image to numpy (для наложения)
    image_np = np.array(image.resize((ckpt["img_size"], ckpt["img_size"]))) / 255.0
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print(
        f"[Grad-CAM] pred_class={pred_class} conf={pred_conf:.4f} target_class={target_class} | saved: {output_path}"
    )
