import torch
import pandas as pd

from src.data.dataset import VarroaDataset, get_transforms
from src.models.classifier import build_model
from src.training.train import get_dataloaders
from src.config import BEST_MODEL_PATH, WEIGHTED_MODEL_PATH, UNFROZEN_MODEL_PATH, METADATA_FILE

def confusion_counts(predictions, labels):
    """Count the four confusion-matrix cells for binary predictions.

    confusion_counts(tensor([1,1,0,0,1]), tensor([1,0,0,1,1])) -> (2, 1, 1, 1)
    """
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    return tp, tn, fp, fn


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall, F1 and the raw counts.

    compute_metrics(tensor([1,1,0,0,1]), tensor([1,0,0,1,1]))
        -> {"accuracy": 0.6, "precision": 0.667, "recall": 0.667, "f1": 0.667, ...}
    """
    tp, tn, fp, fn = confusion_counts(predictions, labels)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def collect_predictions(model, loader, device):
    """Run the model over a loader and return all predictions and labels.

    collect_predictions(model, val_loader, "cpu") -> (tensor [1876], tensor [1876])
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels)

    return torch.cat(all_predictions), torch.cat(all_labels)

def collect_probabilities(model, loader, device):
    """Run the model over a loader and return P(infected) for every image.

    collect_probabilities(model, val_loader, "cpu") -> (tensor [1876] in [0,1], tensor [1876])
    """
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_probabilities.append(probabilities[:, 1].cpu())
            all_labels.append(labels)

    return torch.cat(all_probabilities), torch.cat(all_labels)

def load_checkpoint(path=BEST_MODEL_PATH, device="cpu"):
    """Rebuild the model and load saved weights from a checkpoint file.

    load_checkpoint() -> (model in eval mode, {"epoch": 10, "val_loss": 0.3782, ...})
    """
    checkpoint = torch.load(path, map_location=device)

    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, checkpoint

def per_video_metrics(predictions, labels, split="val"):
    """Compute metrics separately for each source video in a split.

    per_video_metrics(preds, labels, "val") -> {"2017-08-24_...": {"accuracy": 0.87, ...}, ...}
    """
    # Alignment depends on row order: the loader must use shuffle=False and the
    # filtering below must match VarroaDataset.__init__ exactly.
    df = pd.read_csv(METADATA_FILE)
    df = df[df["split"] == split].reset_index(drop=True)

    assert len(df) == len(predictions), f"{len(df)} rows vs {len(predictions)} predictions"

    results = {}

    for video in df["video"].unique():
        mask = torch.tensor((df["video"] == video).values)
        results[video] = compute_metrics(predictions[mask], labels[mask])

    return results

def threshold_sweep(probabilities, labels, thresholds=None):
    """Score the same probabilities at a range of decision thresholds.

    threshold_sweep(probs, labels) -> [{"threshold": 0.05, "recall": 0.91, ...}, ...]
    """
    if thresholds is None:
        thresholds = torch.arange(0.05, 1.0, 0.05)

    results = []

    for threshold in thresholds:
        threshold = float(threshold)
        predictions = (probabilities >= threshold).long()

        metrics = compute_metrics(predictions, labels)
        metrics["threshold"] = threshold
        results.append(metrics)

    return results

if __name__ == "__main__":
    _, val_loader = get_dataloaders()

    for name, path in [("baseline", BEST_MODEL_PATH), ("unfrozen", UNFROZEN_MODEL_PATH)]:
        model, ckpt = load_checkpoint(path)
        probabilities, labels = collect_probabilities(model, val_loader, "cpu")

        print(f"\n=== {name} (epoch {ckpt['epoch']}) ===")
        for m in threshold_sweep(probabilities, labels):
            print(
                f"t={m['threshold']:.2f}  prec {m['precision']:.3f}  "
                f"rec {m['recall']:.3f}  f1 {m['f1']:.3f}  "
                f"fp={m['fp']:4d}  fn={m['fn']:4d}"
            )