from sklearn.preprocessing import StandardScaler
from .config import FEATURE_COLUMNS


def prepare_features(df):
    """Scale numerical audio features for similarity-based ML."""
    X = df[FEATURE_COLUMNS].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler
