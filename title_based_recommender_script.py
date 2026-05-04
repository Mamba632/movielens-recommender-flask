import pandas as pd
import re
from collections import Counter
from math import sqrt
from pathlib import Path

MOVIE_FILE = Path("ml-32m/movies.csv")


def clean_text(text: str) -> str:
    """Lowercase titles and remove punctuation for token matching."""
    text = re.sub(r"\(\d{4}\)$", "", str(text))
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def build_token_counts(title: str) -> Counter:
    return Counter(clean_text(title).split())


def cosine_similarity(a: Counter, b: Counter) -> float:
    intersection = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in intersection)
    norm_a = sqrt(sum(value * value for value in a.values()))
    norm_b = sqrt(sum(value * value for value in b.values()))
    return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


def load_movie_titles(path: Path = MOVIE_FILE) -> pd.DataFrame:
    movies = pd.read_csv(path)
    if "movieId" in movies.columns:
        movies = movies.rename(columns={"movieId": "item_id"})
    movies["clean_title"] = movies["title"].apply(clean_text)
    movies["token_counts"] = movies["title"].apply(build_token_counts)
    return movies


def recommend_similar_movies(query: str, movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    query_tokens = build_token_counts(query)
    scores = []

    for _, row in movies.iterrows():
        score = cosine_similarity(query_tokens, row["token_counts"])
        if score > 0:
            scores.append((row["item_id"], row["title"], score))

    recommendations = sorted(scores, key=lambda item: item[2], reverse=True)
    return pd.DataFrame(recommendations, columns=["item_id", "title", "score"]).head(top_n)


def recommend_for_user(liked_titles: list[str], movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    aggregated = Counter()
    for title in liked_titles:
        aggregated.update(build_token_counts(title))

    scores = []
    for _, row in movies.iterrows():
        if row["title"] in liked_titles:
            continue
        score = cosine_similarity(aggregated, row["token_counts"])
        if score > 0:
            scores.append((row["item_id"], row["title"], score))

    recommendations = sorted(scores, key=lambda item: item[2], reverse=True)
    return pd.DataFrame(recommendations, columns=["item_id", "title", "score"]).head(top_n)


def main() -> None:
    movies = load_movie_titles()
    print(f"Loaded {len(movies)} movies from {MOVIE_FILE.name}")
    print("\nTop recommendations similar to 'Toy Story':")
    print(recommend_similar_movies("Toy Story", movies, top_n=8).to_string(index=False))
    print("\nRecommendations for a user who likes Toy Story and GoldenEye:")
    print(recommend_for_user(["Toy Story (1995)", "GoldenEye (1995)"], movies, top_n=8).to_string(index=False))


if __name__ == "__main__":
    main()
