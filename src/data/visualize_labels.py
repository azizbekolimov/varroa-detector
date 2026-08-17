import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from src.config import RAW_DIR, METADATA_FILE, PROJECT_ROOT

REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def load_metadata():
    """Load metadata.csv with coords kept as a string."""
    df = pd.read_csv(METADATA_FILE, dtype={"coords": str})
    df["coords"] = df["coords"].fillna("")
    return df

def parse_coords(coords_str):
    """Turn a coords string into a list of (x1, y1, x2, y2) tuples."""
    if not coords_str:
        return []
    values = [int(v) for v in coords_str.split(" ")]
    return [tuple(values[i:i + 4]) for i in range(0, len(values), 4)]

def plot_label_grid(df, label, n=20, seed=42, out_path=None):
    """Plot a 5-column grid of sample images for one label, with boxes drawn."""
    subset = df[df["label"] == label]
    sample = subset.sample(n=min(n, len(subset)), random_state=seed)

    n_cols = 5
    n_rows = -(-len(sample) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        image = Image.open(RAW_DIR / row["path"])
        ax.imshow(image)
        for x1, y1, x2, y2 in parse_coords(row["coords"]):
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1)
            ax.add_patch(rect)
        ax.set_title(row["bee_key"], fontsize=6)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()

def main():
    df = load_metadata()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for label in [0, 1, 3]:
        out_path = REPORTS_FIGURES_DIR / f"label_{label}_samples.png"
        plot_label_grid(df, label, out_path=out_path)
        print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
