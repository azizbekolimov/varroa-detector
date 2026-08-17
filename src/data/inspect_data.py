import pandas as pd
from src.config import GT_FILE, METADATA_FILE, PROCESSED_DIR

def parse_filename(filename):
    """Extract (bee_id, frame) from a VarroaDataset image filename.

    '2017-09-01_10-54-26.mp4-bee_id_7741-15-1.png' -> (7741, 15)
    """
    tail = filename.split(".mp4-")[1].removesuffix(".png")
    parts = tail.split("-")
    bee_id = int(parts[0].removeprefix("bee_id_"))
    frame = int(parts[1])
    return bee_id, frame

def parse_line(line):
    """Parse one gt.csv line into a record dict. Returns None for blank lines."""
    line = line.strip()
    if not line:
        return None

    parts = line.split(" ")
    path = parts[0]
    label = int(parts[1])
    n_boxes = len(parts[2:]) // 4

    path_parts = path.split("/")
    split = path_parts[0]
    video = path_parts[2]
    filename = path_parts[3]
    bee_id, frame = parse_filename(filename)

    return {
        "path": path,
        "split": split,
        "video": video,
        "filename": filename,
        "bee_id": bee_id,
        "frame": frame,
        "bee_key": f"{video}_{bee_id}",
        "label": label,
        "n_boxes": n_boxes,
        "coords": " ".join(parts[2:]),
    }

def load_gt(gt_path):
    """Load gt.csv into a DataFrame with one row per image."""
    records = []
    with open(gt_path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                records.append(parsed)
    return pd.DataFrame(records)    

def _report_overlap(df, column):
    """Print pairwise split overlap for one column."""
    print(f"--- overlap by {column} ---")

    sets = {}
    for name in ["train", "test", "val"]:
        sets[name] = set(df[df["split"] == name][column])
        print(f"{name}: {len(sets[name])} unique")

    for a, b in [("train", "test"), ("train", "val"), ("test", "val")]:
        shared = sets[a] & sets[b]
        print(f"{a} & {b}: {len(shared)}")
        if shared:
            print(f"  overlapping ({len(shared)}): {sorted(shared)[:20]} ...")


def check_overlap(df):
    """Print pairwise overlap between splits for video, bee_id and bee_key."""
    _report_overlap(df, "video")
    _report_overlap(df, "bee_id")
    _report_overlap(df, "bee_key")


def summarize(df):
    """Print descriptive statistics for the parsed dataset."""
    pd.set_option("display.max_rows", None)

    print("--- rows per split ---")
    print(df["split"].value_counts())

    print("--- images per video ---")
    print(df.groupby("video").size())

    print("--- label by split ---")
    print(pd.crosstab(df["split"], df["label"]))

    print("--- label by video ---")
    print(pd.crosstab(df["video"], df["label"]))

    print("--- n_boxes distribution ---")
    print(df["n_boxes"].value_counts().sort_index())

    print("--- frames per bee ---")
    print(df.groupby("bee_key").size().describe())


def main():
    df = load_gt(GT_FILE)
    print(df.shape)
    check_overlap(df)
    summarize(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_FILE, index=False)
    print(f"wrote {len(df)} rows to {METADATA_FILE}")


if __name__ == "__main__":
    main()