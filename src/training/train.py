import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    SEED,
    BEST_MODEL_PATH,
    WEIGHTED_MODEL_PATH,
    UNFROZEN_MODEL_PATH,
    METADATA_FILE,
)
from src.data.dataset import VarroaDataset, get_transforms
from src.models.classifier import build_model


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=2):
    """Build the train and validation DataLoaders.

    get_dataloaders(batch_size=32) -> (loader of 258 batches, loader of 59 batches)
    """
    train_dataset = VarroaDataset("train", get_transforms("train"))
    val_dataset = VarroaDataset("val", get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one pass over the training set and update the model's weights.

    train_one_epoch(model, train_loader, criterion, optimizer, "cpu") -> 0.5123
    """
    model.train()
    model.features.eval()

    running_loss = 0.0
    seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / seen


def validate(model, loader, criterion, device):
    """Run one pass over the validation set without updating the model.

    validate(model, val_loader, criterion, "cpu") -> (0.4821, 0.8103)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            seen += batch_size

    return running_loss / seen, correct / seen


def save_checkpoint(model, epoch, val_loss, val_acc, path=BEST_MODEL_PATH):
    """Save the model's weights and the metrics that produced them.

    save_checkpoint(model, 3, 0.4821, 0.8103) -> writes models/checkpoints/best_model.pt
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    torch.save(checkpoint, path)


def compute_class_weights(split="train"):
    """Inverse-frequency class weights for CrossEntropyLoss.

    compute_class_weights("train") -> tensor([0.7252, 1.6102])
    """
    df = pd.read_csv(METADATA_FILE)
    df = df[df["split"] == split]

    # Same mapping as VarroaDataset: labels 1 and 3 are both "infected".
    binary = (df["label"] > 0).astype(int)

    total = len(df)
    counts = [(binary == 0).sum(), (binary == 1).sum()]
    assert sum(counts) == total

    weights = [total / (2 * c) for c in counts]

    return torch.tensor(weights, dtype=torch.float32)


def main(
    num_epochs=15,
    learning_rate=1e-3,
    backbone_lr=1e-4,
    class_weights=None,
    checkpoint_path=BEST_MODEL_PATH,
    unfreeze_last_n=0,
):
    """Train the model and save the checkpoint with the lowest validation loss.

    main(num_epochs=15) -> prints per-epoch metrics, writes models/checkpoints/best_model.pt
    """
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders()

    model = build_model(unfreeze_last_n=unfreeze_last_n)
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Head is randomly initialised and needs large steps; unfrozen backbone
    # weights are already good and need gentle ones.
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    optimizer = torch.optim.Adam(param_groups)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (unfreeze_last_n={unfreeze_last_n})")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch, val_loss, val_acc, path=checkpoint_path)
            print(f"  saved (val loss {val_loss:.4f})")

    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main(
        num_epochs=15,
        unfreeze_last_n=1,
        checkpoint_path=UNFROZEN_MODEL_PATH,
    )