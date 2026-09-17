import pandas as pd
from .config import DATA_PATH, FEATURE_COLUMNS

REQUIRED_COLUMNS = ["song_id", "title", "artist", "genre", *FEATURE_COLUMNS, "popularity"]


def load_songs(path=DATA_PATH):
    """Load and validate the song dataset."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df["song_id"].duplicated().any():
        raise ValueError("song_id values must be unique.")
    return df
