def precision_at_k(recommended_genres, target_genre, k):
    """Simple relevance metric: songs sharing the query song genre are relevant."""
    items = list(recommended_genres)[:k]
    if not items:
        return 0.0
    relevant = sum(1 for genre in items if genre == target_genre)
    return relevant / len(items)


def evaluate_recommender(df, recommender, k=5):
    scores = []
    for _, row in df.iterrows():
        recs = recommender.recommend_cosine(row["title"], top_n=k)
        scores.append(precision_at_k(recs["genre"], row["genre"], k))
    return {
        "songs_evaluated": len(scores),
        "k": k,
        "mean_precision_at_k": round(sum(scores) / len(scores), 4),
    }
