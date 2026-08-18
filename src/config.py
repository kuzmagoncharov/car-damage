from pathlib import Path
import sys


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass


ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "split"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_PATH = MODELS_DIR / "best.pt"

CLASSES = ("damaged", "normal")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
ARCH = "mobilenet_v3_small"
NUM_WORKERS = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
