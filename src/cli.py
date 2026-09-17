import argparse
from .data_loader import load_songs
from .preprocessor import prepare_features
from .recommender import SongRecommender
from .evaluator import evaluate_recommender


def build_system():
    df = load_songs()
    features, _ = prepare_features(df)
    return df, SongRecommender(df, features)


def print_table(df):
    if df.empty:
        print("No results found.")
    else:
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="AI/ML Song Recommendation System")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List songs in the dataset")
    list_p.add_argument("--genre", help="Optional genre filter")

    rec_p = sub.add_parser("recommend", help="Recommend songs similar to a selected song")
    rec_p.add_argument("--song", required=True, help="Exact song title")
    rec_p.add_argument("--top", type=int, default=5, help="Number of recommendations")
    rec_p.add_argument("--method", choices=["cosine", "knn"], default="cosine")

    sub.add_parser("evaluate", help="Evaluate recommender using Precision@5")

    args = parser.parse_args()
    df, recommender = build_system()

    if args.command == "list":
        view = df[["title", "artist", "genre", "popularity"]]
        if args.genre:
            view = view[view["genre"].str.casefold() == args.genre.casefold()]
        print_table(view.reset_index(drop=True))
    elif args.command == "recommend":
        if args.top < 1:
            parser.error("--top must be at least 1")
        try:
            if args.method == "knn":
                result = recommender.recommend_knn(args.song, args.top)
            else:
                result = recommender.recommend_cosine(args.song, args.top)
            print(f"\nRecommendations for: {args.song} ({args.method})\n")
            print_table(result)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.command == "evaluate":
        metrics = evaluate_recommender(df, recommender, k=5)
        print("Evaluation Results")
        print(f"Songs evaluated       : {metrics['songs_evaluated']}")
        print(f"K                     : {metrics['k']}")
        print(f"Mean Precision@K      : {metrics['mean_precision_at_k']}")


if __name__ == "__main__":
    main()
