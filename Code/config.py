"""Percorsi, costanti e ricette di addestramento."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Skin cancer"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = ROOT / "Model"
RESULTS_DIR = ROOT / "Results"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
REPORT_DIR = RESULTS_DIR / "reports"
DATASET_DIR = RESULTS_DIR / "dataset"

for _d in (MODEL_DIR, FIG_DIR, LOG_DIR, REPORT_DIR, DATASET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ["benign", "malignant"]       # benign = 0, malignant = 1 (classe positiva)

SEED = 42
VAL_FRACTION = 0.15
NUM_WORKERS = 4

# Statistiche ImageNet, obbligatorie con backbone pre-addestrati.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 3
SELECTION_METRIC = "auroc"

# Sotto questa confidenza il predictor si astiene invece di rispondere.
ABSTENTION_THRESHOLD = 0.60

DEFAULT_ARCH = "resnet18"

# Ogni architettura porta con se' gli iperparametri con cui l'ho addestrata,
# cosi' `train.py --arch NOME` riproduce esattamente la run che ho tenuto.
RECIPES = {
    "cnn_v1": {
        "img_size": 128,                # obbligata: il primo layer lineare e' fisso su 512*4*4
        "batch_size": 16,
        "lr": 5e-4,
        "epochs": 60,
        "patience": 12,
        "scheduler": "plateau",
        "augment": False,
        "normalize": False,
    },
    "resnet18": {
        "img_size": 224,
        "batch_size": 32,
        "lr": 1e-4,                     # basso: i pesi ImageNet vanno rifiniti, non riscritti
        "epochs": 30,
        "patience": 10,
        "scheduler": "onecycle",
        "augment": True,
        "normalize": True,
    },
}


def device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
