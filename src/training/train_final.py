import torch
import torch.nn as nn

from src.config import FINAL_MODEL_PATH, SEED
from src.evaluation.cross_validate import build_session_map, fold_dataloaders
from src.models.classifier import build_model
from src.training.train import save_checkpoint, train_one_epoch, validate


# Locked before the test set was touched.
# Held-out session provides the early-stopping signal only; it is not a
# performance estimate. The performance estimate is the grouped-CV result
# (F1 0.654 +/- 0.138). FINAL_THRESHOLD is the median of the five per-fold
# thresholds chosen during cross-validation (0.10, 0.20, 0.40, 0.45, 0.70).
HOLDOUT_SESSION = "2017-08-30_15-42-59"
FINAL_THRESHOLD = 0.40


def main(num_epochs=15, learning_rate=1e-3, backbone_lr=1e-4, unfreeze_last_n=1):
    """Train the final model on 9 sessions, holding one out for checkpointing.

    main(num_epochs=15) -> prints per-epoch metrics, writes models/checkpoints/final_model.pt
    """
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    session_map = build_session_map()
    assert HOLDOUT_SESSION in session_map, f"{HOLDOUT_SESSION} not found"

    train_loader, val_loader = fold_dataloaders([HOLDOUT_SESSION], session_map)
    print(f"Holdout session: {HOLDOUT_SESSION}")
    print(f"Train {len(train_loader.dataset)}  holdout {len(val_loader.dataset)}")
    print(f"Committed threshold: {FINAL_THRESHOLD}")

    model = build_model(unfreeze_last_n=unfreeze_last_n).to(device)
    criterion = nn.CrossEntropyLoss()

    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    optimizer = torch.optim.Adam(param_groups)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}\n")

    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, epoch, val_loss, val_acc, path=FINAL_MODEL_PATH)
            print(f"  saved (val loss {val_loss:.4f})")

    print(f"\nBest epoch {best_epoch}, val loss {best_val_loss:.4f}")
    print(f"Saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main(num_epochs=15)