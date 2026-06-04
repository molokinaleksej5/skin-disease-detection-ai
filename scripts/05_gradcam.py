from pathlib import Path
from src.interpretability.gradcam import generate_gradcam
from src.config import CFG

if __name__ == "__main__":
    image_path = CFG.img_dir / "ISIC_0000001.jpg"
    output_path = CFG.reports_dir / "gradcam" / "example_cam.jpg"

    generate_gradcam(image_path, output_path)
