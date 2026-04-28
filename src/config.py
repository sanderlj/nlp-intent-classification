"""Project configuration for Assignment 1."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TOKENIZATION_DIR = OUTPUTS_DIR / "tokenization_examples"
ERROR_ANALYSIS_DIR = OUTPUTS_DIR / "error_analysis"

# Update these once you choose your two assignment languages.
LANGUAGES = ["amh", "zul"]

RANDOM_SEED = 42

BPE_MERGE_VALUES = [100, 200, 300, 500, 750]
NB_ALPHA_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
LR_C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
CHAR_NGRAM_RANGES = [(1, 1), (2, 3), (2, 4), (3, 5)]


def ensure_directories() -> None:
    """Create expected project directories if they do not already exist."""
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        TOKENIZATION_DIR,
        ERROR_ANALYSIS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
