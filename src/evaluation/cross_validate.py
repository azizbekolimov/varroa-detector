import copy
import random
import statistics

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, METADATA_FILE, SEED
from src.data.dataset import VarroaDataset, get_transforms
from src.evaluation.metrics import collect_probabilities, threshold_sweep
from src.models.classifier import build_model
from src.training.train import train_one_epoch, validate


# 2017-09-25_16-03-38 and ...-2 are one recording session split into two files:
# identical timestamp, and contiguous bee_id ranges (228-1215, then 1216-2238).
# They must never land on opposite sides of a split.
SESSION_ALIASES = {
    "2017-09-25_16-03-38-2": "2017-09-25_16-03-38",
}


def session_of(video):
    """Map a video name to its recording session.

    session_of("2017-09-25_16-03-38-2") -> "2017-09-25_16-03-38"
    session_of("2017-08-30_15-42-59")   -> "2017-08-30_15-42-59"
    """
    return SESSION_ALIASES.get(video, video)


def build_session_map(split_exclude="test"):
    """Group non-excluded videos by recording session.

    build_session_map() -> {"2017-09-25_16-03-38": ["2017-09-25_16-03-38", "...-2"], ...}
    """
    df = pd.read_csv(METADATA_FILE)
    videos = sorted(df[df["split"] != split_exclude]["video"].unique())

    sessions = {}
    for video in videos:
        sessions.setdefault(session_of(video), []).append(video)

    return sessions


def verify_test_isolation():
    """Check that no test video shares a recording session with train or val.

    verify_test_isolation() -> prints test video summary, raises if a session overlaps
    """
    df = pd.read_csv(METADATA_FILE)

    ranges = {}
    for video, sub in df.groupby("video"):
        ranges[video] = {
            "split": sub["split"].iloc[0],
            "n": len(sub),
            "min_id": int(sub["bee_id"].min()),
            "max_id": int(sub["bee_id"].max()),
        }

    test_videos = [v for v, r in ranges.items() if r["split"] == "test"]
    other_videos = [v for v, r in ranges.items() if r["split"] != "test"]

    # Hard check: known session aliases must not straddle the test boundary.
    overlap = {session_of(v) for v in test_videos} & {session_of(v) for v in other_videos}
    assert not overlap, f"test session also appears outside test: {overlap}"

    # Heuristic: contiguous bee_id ranges are how the -2 pair was detected.
    # Adjacency alone is not proof; check the dates before acting on a hit.
    print("Adjacent bee_id ranges crossing the test boundary:")
    found = False
    for t in test_videos:
        for o in other_videos:
            a, b = ranges[t], ranges[o]
            if a["max_id"] + 1 == b["min_id"] or b["max_id"] + 1 == a["min_id"]:
                found = True
                print(f"  {t} ({a['min_id']}-{a['max_id']}) <-> {o} ({b['min_id']}-{b['max_id']})")
    if not found:
        print("  none")

    print("\nTest videos:")
    for v in sorted(test_videos):
        r = ranges[v]
        print(f"  {v:35s} n={r['n']:5d}  bee_id {r['min_id']:5d}-{r['max_id']:5d}")

    print("\nNo session overlap between test and train/val.")


def make_folds(sessions, k=5):
    """Partition session names into k groups for grouped cross-validation.

    make_folds(["a","b","c","d","e"], k=2) -> [["c","e","a"], ["b","d"]]
    """
    sessions = sorted(sessions)
    random.Random(SEED).shuffle(sessions)

    folds = [[] for _ in range(k)]

    for i, session in enumerate(sessions):
        folds[i % k].append(session)

    assert all(len(f) > 0 for f in folds), "k is larger than the number of sessions"
    assert sum(len(f) for f in folds) == len(sessions)

    return folds


def fold_dataloaders(val_sessions, session_map, batch_size=BATCH_SIZE, num_workers=2):
    """Build train/val loaders for one fold, split by whole recording sessions.

    fold_dataloaders(["2017-09-25_16-03-38"], session_map) -> (train_loader, val_loader)
    """
    val_videos = [v for s in val_sessions for v in session_map[s]]
    train_videos = [
        v for s, vids in session_map.items() if s not in val_sessions for v in vids
    ]

    # The whole point of grouped CV: no video may appear on both sides.
    assert not (set(train_videos) & set(val_videos)), "session leaked across the split"

    train_dataset = VarroaDataset(videos=train_videos, transform=get_transforms("train"))
    val_dataset = VarroaDataset(videos=val_videos, transform=get_transforms("val"))

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


def run_fold(train_loader, val_loader, num_epochs=10, learning_rate=1e-3,
             backbone_lr=1e-4, unfreeze_last_n=1, device="cpu"):
    """Train one fold and return its best-F1 metrics on that fold's validation set.

    run_fold(train_loader, val_loader) -> {"f1": 0.68, "threshold": 0.35, "best_epoch": 5, ...}
    """
    torch.manual_seed(SEED)

    model = build_model(unfreeze_last_n=unfreeze_last_n).to(device)
    criterion = nn.CrossEntropyLoss()

    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    optimizer = torch.optim.Adam(param_groups)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"    epoch {epoch:2d}/{num_epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)

    probabilities, labels = collect_probabilities(model, val_loader, device)
    sweep = threshold_sweep(probabilities, labels)
    best = max(sweep, key=lambda m: m["f1"])

    best["best_epoch"] = best_epoch
    best["val_loss"] = best_val_loss
    best["n_val"] = len(labels)
    best["n_pos"] = int(labels.sum())

    return best


def main(k=5, num_epochs=10):
    session_map = build_session_map()
    folds = make_folds(list(session_map.keys()), k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{len(session_map)} sessions, {k} folds, {num_epochs} epochs each\n")

    results = []

    for i, fold in enumerate(folds):
        train_loader, val_loader = fold_dataloaders(fold, session_map)
        print(
            f"fold {i}: train {len(train_loader.dataset):5d}  "
            f"val {len(val_loader.dataset):5d}  sessions {fold}"
        )

        metrics = run_fold(train_loader, val_loader, num_epochs=num_epochs, device=device)
        results.append(metrics)

        print(
            f"  -> epoch {metrics['best_epoch']}  t={metrics['threshold']:.2f}  "
            f"prec {metrics['precision']:.3f}  rec {metrics['recall']:.3f}  "
            f"f1 {metrics['f1']:.3f}\n"
        )

    print("=" * 78)
    print(f"{'fold':>4}  {'n_val':>6} {'pos%':>6} {'ep':>3} {'t':>5} "
          f"{'prec':>6} {'rec':>6} {'f1':>6}")
    for i, m in enumerate(results):
        print(
            f"{i:>4}  {m['n_val']:>6} {m['n_pos'] / m['n_val']:>5.1%} "
            f"{m['best_epoch']:>3} {m['threshold']:>5.2f} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
        )

    print("-" * 78)
    for key in ["precision", "recall", "f1"]:
        values = [m[key] for m in results]
        print(f"{key:>10}: {statistics.mean(values):.3f} "
              f"± {statistics.stdev(values):.3f}  "
              f"(min {min(values):.3f}, max {max(values):.3f})")


if __name__ == "__main__":
    verify_test_isolation()