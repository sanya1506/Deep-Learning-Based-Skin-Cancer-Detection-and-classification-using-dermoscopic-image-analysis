"""Central configuration: paths, hyperparameters, class names and label mappings.

All paths use pathlib so the project runs unchanged on macOS, Linux and Windows.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SAMPLE_DIR = ROOT_DIR / "sample_images"
DOCS_DIR = ROOT_DIR / "docs"

METADATA_CSV_NAME = "HAM10000_metadata.csv"
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# --------------------------------------------------------------------------- classes
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]
BINARY_MAP = {c: ("Malignant" if c in MALIGNANT_CLASSES else "Benign") for c in CLASS_NAMES}
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HAIR_KERNEL_SIZE = 17          # blackhat structuring element size
HAIR_THRESHOLD = 12            # threshold on blackhat response to build the hair mask
HAIR_MIN_LENGTH = 40           # keep only long, thin mask components (hairs), not skin texture
INPAINT_RADIUS = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# --------------------------------------------------------------------------- splits
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42

# --------------------------------------------------------------------------- training
AVAILABLE_MODELS = ["efficientnet_b0", "efficientnet_b0_long", "resnet50", "densenet121",
                    "convnext_tiny", "swin_t"]
DEFAULT_MODEL = "efficientnet_b0"
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
MIN_EPOCHS = 10                # early stopping can only trigger after this many epochs
NUM_WORKERS = 0                # 0 avoids multiprocessing issues on Windows/macOS
QUICK_SAMPLES_PER_CLASS = 40   # --quick mode subset size
QUICK_EPOCHS = 2


def get_device():
    """CUDA if available, else Apple MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_path(model_name: str, quick: bool = False) -> Path:
    """Checkpoint location, shared by src/train.py and the training notebooks."""
    return MODELS_DIR / f"{model_name}{'_quick' if quick else ''}.pth"


# --------------------------------------------------------------------------- full study (report)
import os as _os

HAM_MODELS = AVAILABLE_MODELS                      # the six HAM10000 models of Table 6.1
ARCH_OF = {"efficientnet_b0_long": "efficientnet_b0"}   # run name -> torchvision architecture
CNN_SETTINGS = dict(lr=3e-4, weight_decay=1e-4, schedule="plateau", epochs=15, patience=4,
                    min_epochs=10, batch_size=32, label_smoothing=0.05, dropout=0.3)
TRANSFORMER_SETTINGS = dict(lr=1e-4, weight_decay=0.05, schedule="cosine", epochs=25, patience=6,
                            min_epochs=0, batch_size=32, label_smoothing=0.05, dropout=0.3)
TRAIN_SETTINGS = {
    "efficientnet_b0": CNN_SETTINGS,
    "efficientnet_b0_long": {**CNN_SETTINGS, "epochs": 30, "patience": 6},
    "resnet50": CNN_SETTINGS,
    "densenet121": CNN_SETTINGS,
    "convnext_tiny": TRANSFORMER_SETTINGS,
    "swin_t": TRANSFORMER_SETTINGS,
}
SEEDS = [0, 1, 2]                                  # every study is repeated with these seeds
STUDY_ARCH = "efficientnet_b0"                     # supporting studies use EfficientNet-B0
IMBALANCE_STRATEGIES = ["none", "sqrt", "inverse", "focal", "sampler"]
ENSEMBLE_RANDOM_VECTORS = 4000
N_BOOTSTRAP = 1000
CROP_MARGIN = 0.15
CROP_MIN_AREA = 0.01
SEG_SIZE = 256
EXPLAIN_IMAGES = 200
EXPLAIN_STEPS = 20
GATE_NEOPLASM_RECALL = 0.90
CORRUPTIONS = {
    "brightness": [0.85, 0.70, 0.55, 0.40, 0.30],
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 3.0],
    "jpeg_quality": [80, 60, 40, 20, 10],
}

CONFIGS_DIR = ROOT_DIR / "configs"
DERMACON_DIR = DATA_DIR / "dermacon"               # put the DermaCon-IN download here (any layout)
DDI_DIR = DATA_DIR / "ddi"                         # images + ddi_metadata.csv (optional)
MASK_DIR = DATA_DIR / "ham_masks"                  # HAM10000 lesion masks ISIC_xxxxxxx_segmentation.png
PROCESSED_CROP_DIR = DATA_DIR / "processed_crop"
PROCESSED_DERMACON_DIR = DATA_DIR / "processed_dermacon"
PROCESSED_DERMACON_CROP_DIR = DATA_DIR / "processed_dermacon_crop"
STUDY_DIR = RESULTS_DIR / "study"                  # every number of the report is written below here
PREDS_DIR = RESULTS_DIR / "preds"

# SKIN_SMOKE=1 -> tiny subsets, 1 epoch, few bootstrap samples: checks every code path in minutes
SMOKE = _os.environ.get("SKIN_SMOKE", "0") == "1"
if SMOKE:
    N_BOOTSTRAP = 50
    ENSEMBLE_RANDOM_VECTORS = 50
    EXPLAIN_IMAGES = 8

for _d in (CONFIGS_DIR, DERMACON_DIR, DDI_DIR, MASK_DIR, PROCESSED_CROP_DIR, PROCESSED_DERMACON_DIR,
           PROCESSED_DERMACON_CROP_DIR, STUDY_DIR, PREDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

for _d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, SAMPLE_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
