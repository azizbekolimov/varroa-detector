"""Generate README figures: threshold sweep, PR curve, CV folds, confusion matrix.

Figures 1-2 load FINAL_MODEL_PATH and run inference on the test split.
Figures 3-4 use fixed, hardcoded numbers from prior runs (grouped 5-fold CV
and the sealed test-set evaluation at the committed threshold) and do not
touch the model.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, FINAL_MODEL_PATH, PLOTS_DIR
from src.data.dataset import VarroaDataset, get_transforms
from src.evaluation.metrics import collect_probabilities, load_checkpoint, threshold_sweep
from src.training.train_final import FINAL_THRESHOLD

# Colorblind-safe categorical slots (blue / orange / aqua), validated for
# adjacent-pair CVD separation. Never paired as red/green.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Fixed results from grouped 5-fold cross-validation (src/evaluation/cross_validate.py).
# Retraining takes ~90 minutes; these are recorded outcomes, not recomputed here.
CV_FOLDS = [
    {"fold": 0, "f1": 0.624, "n_val": 5712, "pos": 0.314, "threshold": 0.40},
    {"fold": 1, "f1": 0.520, "n_val": 2426, "pos": 0.188, "threshold": 0.45},
    {"fold": 2, "f1": 0.658, "n_val": 268, "pos": 0.313, "threshold": 0.20},
    {"fold": 3, "f1": 0.584, "n_val": 1168, "pos": 0.390, "threshold": 0.10},
    {"fold": 4, "f1": 0.883, "n_val": 527, "pos": 0.414, "threshold": 0.70},
]
CV_MEAN_F1 = 0.654
CV_STD_F1 = 0.138

# Fixed test-set confusion counts at the committed threshold (src/evaluation/evaluate_test.py).
TEST_TP = 532
TEST_TN = 2206
TEST_FP = 260
TEST_FN = 410


def _test_probabilities():
    """Load the final model and collect P(infected) over the sealed test set.

    _test_probabilities() -> (probs tensor [N], labels tensor [N])
    """
    device = torch.device("cpu")

    test_dataset = VarroaDataset("test", get_transforms("val"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, _ = load_checkpoint(FINAL_MODEL_PATH, device)

    return collect_probabilities(model, test_loader, device)


def plot_threshold_sweep(probabilities, labels, save_path):
    """Plot precision and recall vs threshold on the test set.

    plot_threshold_sweep(probs, labels, PLOTS_DIR / "threshold_sweep.png") -> saved path
    """
    sweep = threshold_sweep(probabilities, labels)
    thresholds = [m["threshold"] for m in sweep]
    precision = [m["precision"] for m in sweep]
    recall = [m["recall"] for m in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, precision, color=BLUE, marker="o", markersize=4, linewidth=2, label="Precision")
    ax.plot(thresholds, recall, color=ORANGE, marker="o", markersize=4, linewidth=2, label="Recall")
    ax.axvline(FINAL_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(
        FINAL_THRESHOLD + 0.01, 0.03, f"committed t={FINAL_THRESHOLD:.2f}",
        color=MUTED, fontsize=10, rotation=90, va="bottom",
    )

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Decision threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision & Recall vs Threshold (test set)", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_pr_curve(probabilities, labels, save_path):
    """Plot precision vs recall on the test set, annotating the t=0.40 point.

    plot_pr_curve(probs, labels, PLOTS_DIR / "pr_curve.png") -> saved path
    """
    sweep = threshold_sweep(probabilities, labels)
    precision = [m["precision"] for m in sweep]
    recall = [m["recall"] for m in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(recall, precision, color=BLUE, marker="o", markersize=4, linewidth=2)

    committed = min(sweep, key=lambda m: abs(m["threshold"] - FINAL_THRESHOLD))
    ax.scatter([committed["recall"]], [committed["precision"]], color=ORANGE, s=90, zorder=5,
               label=f"t={FINAL_THRESHOLD:.2f} (committed)")
    ax.annotate(
        f"t={FINAL_THRESHOLD:.2f}\nP={committed['precision']:.2f}, R={committed['recall']:.2f}",
        xy=(committed["recall"], committed["precision"]),
        xytext=(15, -25), textcoords="offset points",
        fontsize=10, color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1),
    )

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve (test set)", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_cv_folds(save_path):
    """Plot per-fold F1 from grouped 5-fold cross-validation (fixed results).

    plot_cv_folds(PLOTS_DIR / "cv_folds.png") -> saved path
    """
    folds = CV_FOLDS
    labels = [f"fold {f['fold']}" for f in folds]
    f1s = [f["f1"] for f in folds]
    y_pos = range(len(folds))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(y_pos, f1s, color=BLUE, height=0.6, zorder=3)

    ax.axvspan(CV_MEAN_F1 - CV_STD_F1, CV_MEAN_F1 + CV_STD_F1, color=ORANGE, alpha=0.15, zorder=1)
    ax.axvline(CV_MEAN_F1, color=ORANGE, linestyle="--", linewidth=1.5, zorder=2)

    for y, f in zip(y_pos, folds):
        ax.text(f["f1"] + 0.015, y, f"n={f['n_val']:,}", va="center", fontsize=10, color=MUTED)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    # Explicit (reversed) limits put fold 0 at top and leave a margin above it
    # for the mean label, instead of invert_yaxis()'s tight default margins.
    ax.set_ylim(len(folds) - 0.3, -0.9)
    ax.text(CV_MEAN_F1, -0.55, f"mean {CV_MEAN_F1:.3f} ± {CV_STD_F1:.3f}",
             ha="center", va="center", fontsize=10, color=ORANGE)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1 (validation fold)", fontsize=11)
    ax.set_title("Grouped 5-Fold Cross-Validation: per-fold F1", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_confusion_matrix(save_path):
    """Plot the 2x2 test-set confusion matrix at the committed threshold (fixed counts).

    plot_confusion_matrix(PLOTS_DIR / "confusion_matrix.png") -> saved path
    """
    # rows = actual, cols = predicted; order: clean (0), infected (1)
    matrix = [[TEST_TN, TEST_FP], [TEST_FN, TEST_TP]]
    row_totals = [TEST_TN + TEST_FP, TEST_FN + TEST_TP]
    classes = ["clean", "infected"]

    fig, ax = plt.subplots(figsize=(8, 5))

    row_pct = [[matrix[r][c] / row_totals[r] for c in range(2)] for r in range(2)]
    im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)

    for r in range(2):
        for c in range(2):
            pct = row_pct[r][c]
            text_color = "white" if pct > 0.5 else INK
            ax.text(
                c, r, f"{matrix[r][c]:,}\n({pct:.1%})",
                ha="center", va="center", fontsize=13, color=text_color,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (test set, t={FINAL_THRESHOLD:.2f})", fontsize=13)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row percentage", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def main():
    """Generate all four README figures and print their saved paths.

    main() -> writes reports/figures/{threshold_sweep,pr_curve,cv_folds,confusion_matrix}.png
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    probabilities, labels = _test_probabilities()

    saved = [
        plot_threshold_sweep(probabilities, labels, PLOTS_DIR / "threshold_sweep.png"),
        plot_pr_curve(probabilities, labels, PLOTS_DIR / "pr_curve.png"),
        plot_cv_folds(PLOTS_DIR / "cv_folds.png"),
        plot_confusion_matrix(PLOTS_DIR / "confusion_matrix.png"),
    ]

    print("Saved figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
