import pytest
from src.data_loader import load_songs
from src.preprocessor import prepare_features
from src.recommender import SongRecommender


def make_recommender():
    df = load_songs()
    features, _ = prepare_features(df)
    return SongRecommender(df, features)


def test_recommendation_count_and_excludes_query():
    rec = make_recommender()
    result = rec.recommend_cosine("Neon Nights", 5)
    assert len(result) == 5
    assert "Neon Nights" not in result["title"].tolist()


def test_unknown_song_raises_error():
    rec = make_recommender()
    with pytest.raises(ValueError):
        rec.recommend_cosine("Not A Real Song", 5)


def test_knn_returns_recommendations():
    rec = make_recommender()
    result = rec.recommend_knn("Neon Nights", 3)
    assert len(result) == 3
