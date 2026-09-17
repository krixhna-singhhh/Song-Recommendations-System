import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from .config import DEFAULT_TOP_N


class SongRecommender:
    """Content-based recommender using cosine similarity and KNN."""

    def __init__(self, df, feature_matrix):
        self.df = df.reset_index(drop=True).copy()
        self.features = np.asarray(feature_matrix)
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn.fit(self.features)

    def find_song_index(self, title):
        matches = self.df.index[self.df["title"].str.casefold() == title.strip().casefold()].tolist()
        if not matches:
            raise ValueError(f"Song '{title}' not found. Use the list command to see available songs.")
        return matches[0]

    def recommend_cosine(self, title, top_n=DEFAULT_TOP_N):
        idx = self.find_song_index(title)
        scores = cosine_similarity(self.features[idx:idx+1], self.features).flatten()
        order = np.argsort(scores)[::-1]
        order = [i for i in order if i != idx][:top_n]
        result = self.df.iloc[order][["title", "artist", "genre", "popularity"]].copy()
        result["similarity"] = [round(float(scores[i]), 4) for i in order]
        return result.reset_index(drop=True)

    def recommend_knn(self, title, top_n=DEFAULT_TOP_N):
        idx = self.find_song_index(title)
        k = min(top_n + 1, len(self.df))
        distances, indices = self.knn.kneighbors(self.features[idx:idx+1], n_neighbors=k)
        pairs = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if int(i) != idx][:top_n]
        rows = [i for i, _ in pairs]
        result = self.df.iloc[rows][["title", "artist", "genre", "popularity"]].copy()
        result["similarity"] = [round(1.0 - d, 4) for _, d in pairs]
        return result.reset_index(drop=True)
