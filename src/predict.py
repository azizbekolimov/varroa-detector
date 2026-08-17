import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import FINAL_MODEL_PATH
from src.data.dataset import get_transforms
from src.evaluation.metrics import load_checkpoint
from src.training.train_final import FINAL_THRESHOLD


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def predict_image(model, image_path, transform, device="cpu"):
    """Return P(infected) for a single image.

    predict_image(model, "bee.png", transform) -> 0.8312
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probability = torch.softmax(outputs, dim=1)[0, 1]

    return probability.item()


def main():
    parser = argparse.ArgumentParser(description="Detect Varroa mites on bee images.")
    parser.add_argument("path", help="image file or directory of images")
    parser.add_argument("--threshold", type=float, default=FINAL_THRESHOLD)
    parser.add_argument("--checkpoint", default=str(FINAL_MODEL_PATH))
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Not found: {path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    transform = get_transforms("val")

    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"No images in {path}")
            return

        flagged = 0
        probabilities = []

        for image_path in images:
            probability = predict_image(model, image_path, transform, device)
            probabilities.append(probability)

            infected = probability >= args.threshold
            flagged += infected
            label = "INFECTED" if infected else "clean"
            print(f"{image_path.name:50s} {probability:.3f}  {label}")

        mean_probability = sum(probabilities) / len(probabilities)

        print(f"\n{len(images)} images")
        print(f"  flagged at t={args.threshold}: {flagged} ({flagged / len(images):.1%})")
        print(f"  mean P(infected):            {mean_probability:.1%}")
    else:
        probability = predict_image(model, path, transform, device)
        infected = probability >= args.threshold

        print(f"\n{path.name}")
        print(f"  P(infected) = {probability:.3f}")
        print(f"  {'INFECTED' if infected else 'CLEAN'} (threshold {args.threshold})")


if __name__ == "__main__":
    main()