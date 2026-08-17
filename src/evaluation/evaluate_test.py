import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, FINAL_MODEL_PATH
from src.data.dataset import VarroaDataset, get_transforms
from src.evaluation.metrics import (
    collect_probabilities,
    compute_metrics,
    load_checkpoint,
    per_video_metrics,
    threshold_sweep,
)
from src.training.train_final import FINAL_THRESHOLD, HOLDOUT_SESSION


def main():
    """Evaluate the final model on the sealed test set at the committed threshold.

    main() -> prints headline metrics at t=0.40, per-video breakdown, and the full sweep
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = VarroaDataset("test", get_transforms("val"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model, ckpt = load_checkpoint(FINAL_MODEL_PATH, device)

    print(f"Model:     epoch {ckpt['epoch']}, holdout val loss {ckpt['val_loss']:.4f}")
    print(f"Holdout:   {HOLDOUT_SESSION}")
    print(f"Threshold: {FINAL_THRESHOLD} (committed before this run)")
    print(f"Test set:  {len(test_dataset)} images\n")

    probabilities, labels = collect_probabilities(model, test_loader, device)
    predictions = (probabilities >= FINAL_THRESHOLD).long()

    m = compute_metrics(predictions, labels)

    print("=" * 60)
    print(f"TEST RESULT at t={FINAL_THRESHOLD}")
    print("=" * 60)
    print(f"  precision {m['precision']:.3f}")
    print(f"  recall    {m['recall']:.3f}")
    print(f"  f1        {m['f1']:.3f}")
    print(f"  accuracy  {m['accuracy']:.3f}")
    print(f"  tp={m['tp']}  tn={m['tn']}  fp={m['fp']}  fn={m['fn']}")
    print("=" * 60)

    print("\nPer video:")
    for video, vm in per_video_metrics(predictions, labels, split="test").items():
        n = vm["tp"] + vm["tn"] + vm["fp"] + vm["fn"]
        pos = vm["tp"] + vm["fn"]
        print(
            f"  {video:25s} n={n:5d} pos={pos:4d} ({pos / n:5.1%})  "
            f"prec {vm['precision']:.3f}  rec {vm['recall']:.3f}  f1 {vm['f1']:.3f}"
        )

    # Descriptive only. The headline above is the reported result; the operating
    # point was fixed before this run and is not revised based on what follows.
    print("\nFull sweep (descriptive, not used for selection):")
    for s in threshold_sweep(probabilities, labels):
        marker = "  <-- committed" if abs(s["threshold"] - FINAL_THRESHOLD) < 1e-6 else ""
        print(
            f"  t={s['threshold']:.2f}  prec {s['precision']:.3f}  "
            f"rec {s['recall']:.3f}  f1 {s['f1']:.3f}{marker}"
        )


if __name__ == "__main__":
    main()