from pathlib import Path

import pytest

from movie_complete import MovieRecommender

DATA_PATH = Path(__file__).resolve().parents[1] / "ml-32m" / "movies.csv"


def test_load_movies():
    recommender = MovieRecommender(csv_path=DATA_PATH)
    assert len(recommender.movies) > 0
    assert "clean_title" in recommender.movies.columns
    assert "avg_rating" in recommender.movies.columns
    assert "tags_list" in recommender.movies.columns
    assert recommender.title_tfidf.shape[0] == len(recommender.movies)


def test_recommend_similar_movies():
    recommender = MovieRecommender(csv_path=DATA_PATH)
    results = recommender.recommend_similar_movies("Toy Story", top_n=5)
    assert not results.empty
    assert "score" in results.columns
    assert "avg_rating" in results.columns
    assert "num_ratings" in results.columns
    assert all(results["score"] >= 0)


def test_get_closest_titles():
    recommender = MovieRecommender(csv_path=DATA_PATH)
    suggestions = recommender.get_closest_titles("Toy Stoy", n=3)
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


def test_recommend_for_user():
    recommender = MovieRecommender(csv_path=DATA_PATH)
    results = recommender.recommend_for_user(["Toy Story"], top_n=5)
    assert not results.empty
    assert "score" in results.columns
    assert "avg_rating" in results.columns
    assert "num_ratings" in results.columns
    assert all(results["score"] >= 0)
