from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"
DEFAULT_TOP_N = 5
RANDOM_STATE = 42
FEATURE_COLUMNS = ["danceability", "energy", "valence", "tempo", "acousticness"]
