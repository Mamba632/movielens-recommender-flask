import pandas as pd
import re
from collections import Counter
from math import sqrt
from pathlib import Path

MOVIE_FILE = Path("ml-32m/movies.csv")

# Common words to ignore for better recommendations
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'of', 'to', 'is', 'it', 'as', 'for', 'by'}


def clean_text(text: str) -> str:
    """Normalize and clean movie titles."""
    text = re.sub(r"\(\d{4}\)$", "", str(text))
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def build_token_counts(title: str, remove_stopwords: bool = True) -> Counter:
    """Extract meaningful words from cleaned title."""
    tokens = clean_text(title).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return Counter(tokens)


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Fast cosine similarity between two token counts."""
    if not a or not b:
        return 0.0
    
    intersection = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in intersection)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    
    return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


def load_movie_titles(path: Path = MOVIE_FILE) -> pd.DataFrame:
    """Load and preprocess movies with efficient caching."""
    movies = pd.read_csv(path)
    if "movieId" in movies.columns:
        movies = movies.rename(columns={"movieId": "item_id"})
    movies["clean_title"] = movies["title"].apply(clean_text)
    movies["token_counts"] = movies["title"].apply(
        lambda x: build_token_counts(x, remove_stopwords=True)
    )
    return movies


def recommend_similar_movies(
    query: str,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_similarity: float = 0.1
) -> pd.DataFrame:
    """Find movies similar to query with minimum threshold filtering."""
    query_tokens = build_token_counts(query, remove_stopwords=True)
    
    if not query_tokens:
        return pd.DataFrame(columns=["item_id", "title", "score"])
    
    # Vectorized similarity computation
    scores = movies["token_counts"].apply(
        lambda tokens: cosine_similarity(query_tokens, tokens)
    )
    
    # Filter by minimum similarity and get top N
    mask = scores >= min_similarity
    results = movies[mask].copy()
    results["score"] = scores[mask]
    
    return results.nlargest(top_n, "score")[["item_id", "title", "score"]]


def recommend_for_user(
    liked_titles: list[str],
    movies: pd.DataFrame,
    top_n: int = 10,
    min_similarity: float = 0.1
) -> pd.DataFrame:
    """Recommend movies based on user's liked movies."""
    if not liked_titles:
        return pd.DataFrame(columns=["item_id", "title", "score"])
    
    # Combine all liked movie tokens
    aggregated = Counter()
    for title in liked_titles:
        aggregated.update(build_token_counts(title, remove_stopwords=True))
    
    if not aggregated:
        return pd.DataFrame(columns=["item_id", "title", "score"])
    
    # Filter out already-liked movies and compute scores
    user_liked_set = set(liked_titles)
    mask = ~movies["title"].isin(user_liked_set)
    
    scores = movies[mask]["token_counts"].apply(
        lambda tokens: cosine_similarity(aggregated, tokens)
    )
    
    # Filter by minimum similarity and get top N
    similarity_mask = scores >= min_similarity
    results = movies[mask][similarity_mask].copy()
    results["score"] = scores[similarity_mask]
    
    return results.nlargest(top_n, "score")[["item_id", "title", "score"]]


def interactive_recommender():
    """Interactive loop for user recommendations."""
    movies = load_movie_titles()
    print(f"Loaded {len(movies)} movies from {MOVIE_FILE.name}\n")
    
    while True:
        choice = input("Choose: (1) Find similar movies, (2) Recommendations for liked movies, (3) Exit\n> ").strip()
        
        if choice == "1":
            query = input("Enter a movie title: ").strip()
            if query:
                results = recommend_similar_movies(query, movies, top_n=8)
                if results.empty:
                    print("No movies found.\n")
                else:
                    print(f"\nMovies similar to '{query}':")
                    print(results.to_string(index=False))
                    print()
        
        elif choice == "2":
            liked_input = input("Enter movie titles (comma-separated): ").strip()
            liked_titles = [t.strip() for t in liked_input.split(",") if t.strip()]
            if liked_titles:
                results = recommend_for_user(liked_titles, movies, top_n=8)
                if results.empty:
                    print("No recommendations found.\n")
                else:
                    print(f"\nRecommendations based on: {', '.join(liked_titles)}")
                    print(results.to_string(index=False))
                    print()
        
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.\n")


def main() -> None:
    """Run demo with improvements."""
    movies = load_movie_titles()
    print(f"Loaded {len(movies)} movies from {MOVIE_FILE.name}\n")
    
    # Demo 1: Similar movies
    print("=" * 60)
    print("DEMO 1: Movies similar to 'Toy Story'")
    print("=" * 60)
    results = recommend_similar_movies("Toy Story", movies, top_n=8, min_similarity=0.15)
    print(results.to_string(index=False))
    
    # Demo 2: User recommendations
    print("\n" + "=" * 60)
    print("DEMO 2: Recommendations for user who likes Toy Story & GoldenEye")
    print("=" * 60)
    results = recommend_for_user(
        ["Toy Story (1995)", "GoldenEye (1995)"],
        movies,
        top_n=8,
        min_similarity=0.15
    )
    print(results.to_string(index=False))
    
    # Demo 3: Interactive mode
    print("\n" + "=" * 60)
    print("Want to try interactive mode? (y/n)")
    if input("> ").strip().lower() == "y":
        interactive_recommender()


if __name__ == "__main__":
    main()
