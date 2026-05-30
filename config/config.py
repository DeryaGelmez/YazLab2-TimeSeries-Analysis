from pathlib import Path

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# Dataset Paths

SKAB_PROCESSED_DIR = PROCESSED_DATA_DIR / "SKAB"
BATADAL_PROCESSED_DIR = PROCESSED_DATA_DIR / "BATADAL"

SKAB_TRAIN_SCALED = SKAB_PROCESSED_DIR / "skab_train_scaled.csv"
SKAB_TEST_SCALED = SKAB_PROCESSED_DIR / "skab_test_scaled.csv"

SKAB_TRAIN_PC1 = SKAB_PROCESSED_DIR / "skab_train_pc1.csv"
SKAB_TEST_PC1 = SKAB_PROCESSED_DIR / "skab_test_pc1.csv"

BATADAL_TRAIN_SCALED = BATADAL_PROCESSED_DIR / "batadal_train_scaled.csv"
BATADAL_VAL_SCALED = BATADAL_PROCESSED_DIR / "batadal_val_scaled.csv"
BATADAL_TEST_SCALED = BATADAL_PROCESSED_DIR / "batadal_test_scaled.csv"

BATADAL_TRAIN_PC1 = BATADAL_PROCESSED_DIR / "batadal_train_pc1.csv"
BATADAL_VAL_PC1 = BATADAL_PROCESSED_DIR / "batadal_val_pc1.csv"
BATADAL_TEST_PC1 = BATADAL_PROCESSED_DIR / "batadal_test_pc1.csv"

# Experiment Parameters

RANDOM_SEEDS = [42, 123, 2026, 7, 999]

# Deep learning parameters
MAX_EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5

# Automata fixed comparison parameters
DEFAULT_WINDOW_SIZE = 4
DEFAULT_ALPHABET_SIZE = 3

# Automata parameter analysis
WINDOW_SIZES = [3, 4, 5, 6]
ALPHABET_SIZES = [3, 4, 5, 6]

WINDOW_SIZE = 6
ALPHABET_SIZE = 6

# Noise experiment
GAUSSIAN_NOISE_STD = 0.05

# Classification threshold for automata anomaly decision
AUTOMATA_PROBABILITY_THRESHOLD = 0.10

# State diagram gorsel esigi: bu olasiligin altindaki gecisler cizilmez
STATE_DIAGRAM_THRESHOLD = 0.3

# Controlled unseen senaryo: test pattern'larinin yuzde kaci modifiye edilir
UNSEEN_RATIO = 0.10

# Deep learning outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_METRICS_DIR = OUTPUTS_DIR / "metrics"
OUTPUTS_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"

# Deep learning hyperparameters
DL_SEQUENCE_LENGTH = 30
DL_STRIDE = 1
DL_LEARNING_RATE = 1e-3

# DataLoader / GPU throughput (batch size stays 32 per project spec)
DL_NUM_WORKERS = 0
DL_PREFETCH_FACTOR = 2
DL_PIN_MEMORY = True
DL_USE_AMP = True

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
    DL_DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"
    if not _CUDA_AVAILABLE:
        DL_NUM_WORKERS = 0
        DL_PIN_MEMORY = False
        DL_USE_AMP = False
except ImportError:
    DL_DEVICE = "cpu"
    DL_NUM_WORKERS = 0
    DL_PIN_MEMORY = False
    DL_USE_AMP = False

# Model-specific parameters
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2

GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 1
GRU_DROPOUT = 0.2

CNN1D_NUM_FILTERS = 64
CNN1D_KERNEL_SIZE = 3
CNN1D_DROPOUT = 0.3

# Deep learning scenario constants
DL_GAUSSIAN_NOISE_STD = 0.1
DL_UNSEEN_DRIFT_MEAN = 0.3
DL_UNSEEN_DRIFT_SCALE = 1.15
DL_SKAB_N_FOLDS = 5

# Grid 2: imbalance-aware BCE (pos_weight from train window positive rate)
DL_WEIGHTED_BCE_SUFFIX = "_wbce"
DL_POS_WEIGHT_MAX = 25.0


def create_required_dirs():
    for directory in [
        RESULTS_DIR,
        LOGS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        OUTPUTS_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_METRICS_DIR,
        OUTPUTS_MODELS_DIR,
        OUTPUTS_LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)