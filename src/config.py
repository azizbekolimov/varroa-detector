from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
VAL_DIR = RAW_DIR / "val"
GT_FILE = RAW_DIR / "gt.csv"
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "reports" / "figures"


BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
WEIGHTED_MODEL_PATH = CHECKPOINT_DIR / "best_model_weighted.pt"
UNFROZEN_MODEL_PATH = CHECKPOINT_DIR / "best_model_unfrozen.pt"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pt"

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
SEED = 42