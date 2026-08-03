"""Shared project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "data.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Raw column names in data/data.csv -> canonical names used across the project.
COLUMN_MAP = {"Sentence": "text", "Sentiment": "sentiment"}

# Sentiment label <-> integer id.
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

RANDOM_STATE = 42

REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
