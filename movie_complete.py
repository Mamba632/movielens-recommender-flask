import difflib
import re
import time
from collections import Counter
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIE_FILE = Path("ml-32m/movies.csv")
RATINGS_FILE = Path("ml-32m/ratings.csv")
TAGS_FILE = Path("ml-32m/tags.csv")
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'of', 'to', 'is', 'it', 'as', 'for', 'by', 'from', 'at', 'on'}


class MovieRecommender:
    """Production-ready movie recommendation system with advanced features."""
    
    def __init__(self, csv_path: Path = MOVIE_FILE):
        """Initialize with caching for performance."""
        self.csv_path = csv_path
        self.movies = None
        self.stats = {
            'total_recommendations': 0,
            'total_searches': 0,
            'avg_computation_time': 0,
            'load_time': 0
        }
        self.load_time_start = time.time()
        self.load_movies()
    
    def load_movies(self) -> None:
        """Load and preprocess movies with ratings and tags."""
        self.movies = pd.read_csv(self.csv_path)
        if "movieId" in self.movies.columns:
            self.movies = self.movies.rename(columns={"movieId": "item_id"})
        
        # Load ratings and compute aggregates
        if RATINGS_FILE.exists():
            ratings = pd.read_csv(RATINGS_FILE)
            rating_stats = ratings.groupby('movieId').agg(
                avg_rating=('rating', 'mean'),
                num_ratings=('rating', 'count')
            ).reset_index()
            rating_stats = rating_stats.rename(columns={'movieId': 'item_id'})
            self.movies = self.movies.merge(rating_stats, on='item_id', how='left')
            self.movies['avg_rating'] = self.movies['avg_rating'].fillna(0)
            self.movies['num_ratings'] = self.movies['num_ratings'].fillna(0).astype(int)
        else:
            self.movies['avg_rating'] = 0.0
            self.movies['num_ratings'] = 0
        
        # Load tags and aggregate
        if TAGS_FILE.exists():
            tags = pd.read_csv(TAGS_FILE)
            tags['tag'] = tags['tag'].fillna('').astype(str)  # Handle NaN and convert to str
            tag_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(set(str(tag) for tag in x.unique() if tag != ''))).reset_index()
            tag_agg = tag_agg.rename(columns={'movieId': 'item_id', 'tag': 'tags_list'})
            self.movies = self.movies.merge(tag_agg, on='item_id', how='left')
            self.movies['tags_list'] = self.movies['tags_list'].fillna('')
        else:
            self.movies['tags_list'] = ''
        
        # Combine title, genres, and tags for richer text matching
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["combined_text"] = (
            self.movies["title"] + " " + self.movies["genres"] + " " + self.movies["tags_list"]
        )

        self.movies["clean_title"] = self.movies["title"].apply(self._clean_text)
        self.movies["token_counts"] = self.movies["combined_text"].apply(
            lambda x: self._build_token_counts(x, remove_stopwords=True)
        )
        self.movies["year"] = self.movies["title"].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        self.clean_titles = self.movies["clean_title"].tolist()
        self._prepare_tfidf_matrix()
        self.stats['load_time'] = time.time() - self.load_time_start
        print(f"[OK] Loaded {len(self.movies)} movies with ratings and tags in {self.stats['load_time']:.3f}s")

    def _prepare_tfidf_matrix(self) -> None:
        """Build TF-IDF vectors for all movie titles and tags."""
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.title_tfidf = self.tfidf_vectorizer.fit_transform(self.movies["combined_text"])

    def _parse_query_year(self, query: str) -> int:
        """Extract a year from the query text, if present."""
        match = re.search(r"\((\d{4})\)", query)
        return int(match.group(1)) if match else 0

    def _find_best_title_match(self, query: str) -> Optional[str]:
        """Return a best match title from the dataset for a misspelled query."""
        query_clean = self._clean_text(query)
        if query_clean in self.clean_titles:
            return self.movies.loc[self.movies["clean_title"] == query_clean, "title"].iloc[0]

        suggestions = difflib.get_close_matches(query_clean, self.clean_titles, n=1, cutoff=0.45)
        if suggestions:
            return self.movies.loc[self.movies["clean_title"] == suggestions[0], "title"].iloc[0]
        return None

    def get_closest_titles(self, query: str, n: int = 5) -> List[str]:
        """Return close movie titles when exact matches are not found."""
        query_clean = self._clean_text(query)
        if not query_clean:
            return []
        suggestions = difflib.get_close_matches(query_clean, self.clean_titles, n=n, cutoff=0.45)
        return [self.movies.loc[self.movies["clean_title"] == title, "title"].iloc[0] for title in suggestions]

    def _apply_year_boost(self, scores: np.ndarray, query_year: int) -> np.ndarray:
        """Slightly boost movies that match the query year."""
        if query_year == 0:
            return scores

        year_values = self.movies["year"].fillna(0).astype(int).to_numpy()
        boosts = []
        for year in year_values:
            if year > 0:
                diff = abs(year - query_year)
                similarity = max(0.0, 1 - diff / 20)
                boosts.append(0.9 + 0.2 * similarity)
            else:
                boosts.append(1.0)
        return scores * np.array(boosts)

    def _apply_popularity_boost(self, scores: np.ndarray, min_rating: float = 0.0) -> np.ndarray:
        """Boost movies based on average rating and number of ratings."""
        avg_ratings = self.movies['avg_rating'].to_numpy()
        num_ratings = self.movies['num_ratings'].to_numpy()
        
        boosts = []
        for avg, num in zip(avg_ratings, num_ratings):
            if avg >= min_rating and num > 0:
                # Boost based on rating (higher rating = higher boost) and popularity (more ratings = slight boost)
                rating_boost = 0.8 + 0.4 * (avg / 5.0)  # Scale to 0.8-1.2
                popularity_boost = min(1.0 + 0.1 * (num / 1000), 1.2)  # Cap at 1.2
                boosts.append(rating_boost * popularity_boost)
            else:
                boosts.append(0.8)  # Slight penalty for unrated or low-rated
        return scores * np.array(boosts)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize movie titles."""
        text = re.sub(r"\(\d{4}\)$", "", str(text))
        text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()
    
    @staticmethod
    def _build_token_counts(title: str, remove_stopwords: bool = True) -> Counter:
        """Extract meaningful words."""
        tokens = MovieRecommender._clean_text(title).split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return Counter(tokens)
    
    @staticmethod
    def _cosine_similarity(a: Counter, b: Counter) -> float:
        """Compute cosine similarity."""
        if not a or not b:
            return 0.0
        intersection = set(a) & set(b)
        numerator = sum(a[token] * b[token] for token in intersection)
        norm_a = sqrt(sum(v * v for v in a.values()))
        norm_b = sqrt(sum(v * v for v in b.values()))
        return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0
    
    @staticmethod
    def _get_matching_keywords(query_tokens: Counter, movie_tokens: Counter) -> List[str]:
        """Extract matching keywords between two token sets."""
        intersection = set(query_tokens) & set(movie_tokens)
        return sorted(list(intersection))
    
    def recommend_similar_movies(
        self,
        query: str,
        top_n: int = 10,
        min_similarity: float = 0.1,
        year_range: Optional[Tuple[int, int]] = None,
        min_rating: float = 0.0
    ) -> pd.DataFrame:
        """Find similar movies with TF-IDF, year boosting, and popularity boosting."""
        start_time = time.time()
        query_text = self._clean_text(query)
        if not query_text:
            return pd.DataFrame(columns=["item_id", "title", "score", "matching_keywords", "year", "genres", "tags_list", "avg_rating", "num_ratings"])

        query_year = self._parse_query_year(query)
        query_vec = self.tfidf_vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.title_tfidf).flatten()
        scores = self._apply_year_boost(scores, query_year)
        scores = self._apply_popularity_boost(scores, min_rating)

        query_tokens = self._build_token_counts(query, remove_stopwords=True)
        keywords = self.movies["token_counts"].apply(
            lambda tokens: ", ".join(self._get_matching_keywords(query_tokens, tokens))
        )

        mask = scores >= min_similarity
        results = self.movies[mask].copy()
        results["score"] = scores[mask]
        results["matching_keywords"] = keywords[mask]

        if year_range:
            min_year, max_year = year_range
            results = results[(results["year"] >= min_year) & (results["year"] <= max_year)]

        results = results.nlargest(top_n, "score")[["item_id", "title", "score", "matching_keywords", "year", "genres", "tags_list", "avg_rating", "num_ratings"]]

        self.stats['total_searches'] += 1
        elapsed = time.time() - start_time
        self.stats['avg_computation_time'] = (
            (self.stats['avg_computation_time'] * (self.stats['total_searches'] - 1) + elapsed) / 
            self.stats['total_searches']
        )

        return results

    def recommend_for_user(
        self,
        liked_titles: List[str],
        top_n: int = 10,
        min_similarity: float = 0.1,
        year_range: Optional[Tuple[int, int]] = None,
        min_rating: float = 0.0
    ) -> pd.DataFrame:
        """Recommend based on liked movies, with fuzzy matching and TF-IDF profile aggregation."""
        start_time = time.time()
        liked_titles = [title.strip() for title in liked_titles if title.strip()]
        if not liked_titles:
            return pd.DataFrame(columns=["item_id", "title", "score", "matching_keywords", "year", "genres", "tags_list", "avg_rating", "num_ratings"])

        profile_vectors = []
        resolved_titles = []
        for title in liked_titles:
            best_match = self._find_best_title_match(title)
            if best_match:
                resolved_titles.append(best_match)
                idx = self.movies[self.movies["title"] == best_match].index[0]
                profile_vectors.append(self.title_tfidf[idx])
            else:
                profile_vectors.append(self.tfidf_vectorizer.transform([self._clean_text(title)]))

        if not profile_vectors:
            return pd.DataFrame(columns=["item_id", "title", "score", "matching_keywords", "year", "avg_rating", "num_ratings"])

        profile_vector = vstack(profile_vectors).mean(axis=0)
        profile_vector = np.asarray(profile_vector).reshape(1, -1)
        scores = cosine_similarity(profile_vector, self.title_tfidf).flatten()
        scores = self._apply_popularity_boost(scores, min_rating)

        aggregated = Counter()
        for title in liked_titles:
            aggregated.update(self._build_token_counts(title, remove_stopwords=True))

        mask = ~self.movies["title"].isin(resolved_titles)
        similarity_mask = scores >= min_similarity
        candidate_mask = mask.to_numpy() & similarity_mask
        candidate_indices = np.where(candidate_mask)[0]

        results = self.movies.iloc[candidate_indices].copy()
        results["score"] = scores[candidate_indices]
        results["matching_keywords"] = results["token_counts"].apply(
            lambda tokens: ", ".join(self._get_matching_keywords(aggregated, tokens))
        )

        if year_range:
            min_year, max_year = year_range
            results = results[(results["year"] >= min_year) & (results["year"] <= max_year)]

        results = results.nlargest(top_n, "score")[["item_id", "title", "score", "matching_keywords", "year", "genres", "tags_list", "avg_rating", "num_ratings"]]

        self.stats['total_recommendations'] += 1
        elapsed = time.time() - start_time
        self.stats['avg_computation_time'] = (
            (self.stats['avg_computation_time'] * (self.stats['total_recommendations'] - 1) + elapsed) / 
            self.stats['total_recommendations']
        )

        return results
    
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Full-text search in movie titles with fuzzy fallback."""
        query_clean = self._clean_text(query)
        if not query_clean:
            return pd.DataFrame(columns=["item_id", "title", "year"])

        mask = self.movies["clean_title"].str.contains(re.escape(query_clean), regex=True)
        results = self.movies[mask][["item_id", "title", "year"]].head(limit)
        if results.empty:
            close_titles = self.get_closest_titles(query, n=limit)
            return self.movies[self.movies["title"].isin(close_titles)][["item_id", "title", "year"]]
        return results
    
    def get_statistics(self) -> Dict:
        """Return performance and usage statistics."""
        return {
            "total_movies": len(self.movies),
            "total_searches": self.stats['total_searches'],
            "total_recommendations": self.stats['total_recommendations'],
            "avg_computation_time_ms": self.stats['avg_computation_time'] * 1000,
            "data_load_time_s": self.stats['load_time'],
            "timestamp": datetime.now().isoformat()
        }
    
    def export_results(self, results: pd.DataFrame, filename: str, format: str = "csv") -> None:
        """Export recommendations to file."""
        if format == "csv":
            results.to_csv(filename, index=False)
            print(f"[OK] Results exported to {filename}")
        elif format == "json":
            results.to_json(filename, orient="records", indent=2)
            print(f"[OK] Results exported to {filename}")
        else:
            raise ValueError("Format must be 'csv' or 'json'")
    
    def get_year_distribution(self) -> Dict[int, int]:
        """Analyze movie distribution by year."""
        return self.movies[self.movies["year"] > 0]["year"].value_counts().to_dict()
    
    def get_top_keywords(self, n: int = 20) -> Dict[str, int]:
        """Most common keywords across all movies."""
        all_tokens = Counter()
        for token_count in self.movies["token_counts"]:
            all_tokens.update(token_count)
        return dict(all_tokens.most_common(n))
    
    def interactive_mode(self) -> None:
        """Full interactive recommendation engine."""
        print("\n" + "="*70)
        print("INTERACTIVE MOVIE RECOMMENDER")
        print("="*70 + "\n")
        
        while True:
            print("\nOptions:")
            print("1. Find similar movies")
            print("2. Get recommendations for liked movies")
            print("3. Search movies by title")
            print("4. View statistics")
            print("5. View year distribution")
            print("6. View top keywords")
            print("7. Export results")
            print("8. Exit")
            
            choice = input("\nChoose option (1-8): ").strip()
            
            if choice == "1":
                query = input("Enter movie title: ").strip()
                if query:
                    results = self.recommend_similar_movies(query, top_n=8)
                    if results.empty:
                        print("[X] No movies found.")
                    else:
                        print(f"\n[OK] Movies similar to '{query}':")
                        print(results.to_string(index=False))
            
            elif choice == "2":
                liked_input = input("Enter movie titles (comma-separated): ").strip()
                liked_titles = [t.strip() for t in liked_input.split(",") if t.strip()]
                if liked_titles:
                    results = self.recommend_for_user(liked_titles, top_n=8)
                    if results.empty:
                        print("[X] No recommendations found.")
                    else:
                        print(f"\n[OK] Recommendations for: {', '.join(liked_titles)}")
                        print(results.to_string(index=False))
            
            elif choice == "3":
                query = input("Search for movie: ").strip()
                if query:
                    results = self.search_movies(query, limit=10)
                    if results.empty:
                        print("[X] No matches found.")
                    else:
                        print(f"\n[OK] Search results for '{query}':")
                        print(results.to_string(index=False))
            
            elif choice == "4":
                stats = self.get_statistics()
                print("\nSTATISTICS:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            
            elif choice == "5":
                dist = self.get_year_distribution()
                print("\nTOP 15 YEARS BY MOVIE COUNT:")
                for year, count in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:15]:
                    print(f"  {year}: {count} movies")
            
            elif choice == "6":
                keywords = self.get_top_keywords(20)
                print("\nTOP 20 KEYWORDS:")
                for keyword, count in keywords.items():
                    print(f"  {keyword}: {count}")
            
            elif choice == "7":
                query = input("Enter movie title or liked movies (comma-separated): ").strip()
                file_format = input("Format (csv/json): ").strip().lower()
                
                if "," in query:
                    liked = [t.strip() for t in query.split(",")]
                    results = self.recommend_for_user(liked, top_n=20)
                else:
                    results = self.recommend_similar_movies(query, top_n=20)
                
                if not results.empty:
                    filename = f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_format}"
                    self.export_results(results, filename, file_format)
            
            elif choice == "8":
                print("\nThank you for using Movie Recommender!")
                break
            
            else:
                print("[X] Invalid option. Try again.")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = f"""
{'='*70}
{'MOVIE RECOMMENDER SYSTEM - COMPREHENSIVE REPORT':^70}
{'='*70}

DATASET STATISTICS
{'-'*70}
Total Movies:           {len(self.movies)}
Data Load Time:         {self.stats['load_time']:.3f} seconds
Year Range:             {int(self.movies['year'].min())} - {int(self.movies['year'].max())}

USAGE STATISTICS
{'-'*70}
Total Searches:         {self.stats['total_searches']}
Total Recommendations:  {self.stats['total_recommendations']}
Avg Computation Time:   {self.stats['avg_computation_time']*1000:.2f} ms

TOP 10 KEYWORDS
{'-'*70}
"""
        for keyword, count in list(self.get_top_keywords(10).items()):
            report += f"  - {keyword:<20} {count:>5} occurrences\n"
        
        report += f"\nTOP 5 YEARS\n{'-'*70}\n"
        for year, count in sorted(self.get_year_distribution().items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"  - {int(year):<10} {count:>5} movies\n"
        
        report += f"\n[OK] Report generated at {datetime.now().isoformat()}\n"
        return report


def main():
    """Main entry point."""
    print("\nInitializing Movie Recommender System...\n")
    recommender = MovieRecommender()
    
    # Display a comprehensive report
    print(recommender.generate_report())
    
    # Run demo
    print("\n" + "="*70)
    print("DEMO: Similar Movies to 'Toy Story'")
    print("="*70)
    results = recommender.recommend_similar_movies("Toy Story", top_n=8)
    print(results.to_string(index=False))
    
    print("\n" + "="*70)
    print("DEMO: Recommendations for user who likes Toy Story & GoldenEye")
    print("="*70)
    results = recommender.recommend_for_user(
        ["Toy Story (1995)", "GoldenEye (1995)"],
        top_n=8
    )
    print(results.to_string(index=False))
    
    # Ask for interactive mode
    print("\n" + "="*70)
    interactive = input("Launch interactive mode? (y/n): ").strip().lower()
    if interactive == "y":
        recommender.interactive_mode()


if __name__ == "__main__":
    main()
