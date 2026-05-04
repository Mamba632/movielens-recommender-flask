from functools import lru_cache
from io import StringIO

from flask import Flask, Response, render_template, request

from movie_complete import MovieRecommender


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


def year_bounds(recommender: MovieRecommender) -> tuple[int, int]:
    years = recommender.movies.loc[recommender.movies["year"] > 0, "year"]
    return int(years.min()), int(years.max())


def results_payload(results):
    if results.empty:
        return []
    display = results.copy()
    if "score" in display.columns:
        display["score"] = display["score"].map(lambda value: f"{value:.3f}")
    if "avg_rating" in display.columns:
        display["avg_rating"] = display["avg_rating"].map(lambda value: f"{value:.2f}" if value else "N/A")
    if "num_ratings" in display.columns:
        display["num_ratings"] = display["num_ratings"].map(lambda value: f"{int(value):,}")
    return display.to_dict(orient="records")


def parse_int(name: str, default: int) -> int:
    try:
        return int(request.values.get(name, default))
    except (TypeError, ValueError):
        return default


def parse_float(name: str, default: float) -> float:
    try:
        return float(request.values.get(name, default))
    except (TypeError, ValueError):
        return default


@app.context_processor
def inject_globals():
    return {"active_path": request.path}


@app.route("/")
def home():
    recommender = get_recommender()
    min_year, max_year = year_bounds(recommender)
    year_distribution = recommender.get_year_distribution()
    top_years = sorted(year_distribution.items(), key=lambda item: item[1], reverse=True)[:10]
    top_keywords = list(recommender.get_top_keywords(12).items())
    return render_template(
        "home.html",
        recommender=recommender,
        min_year=min_year,
        max_year=max_year,
        top_years=top_years,
        top_keywords=top_keywords,
    )


@app.route("/similar", methods=["GET", "POST"])
def similar():
    recommender = get_recommender()
    min_year, max_year = year_bounds(recommender)
    query = request.values.get("query", "").strip()
    top_n = parse_int("top_n", 8)
    min_similarity = parse_float("min_similarity", 0.1)
    start_year = parse_int("start_year", min_year)
    end_year = parse_int("end_year", max_year)
    min_rating = parse_float("min_rating", 0.0)
    results = []
    suggestions = []

    if query:
        frame = recommender.recommend_similar_movies(
            query,
            top_n=top_n,
            min_similarity=min_similarity,
            year_range=(start_year, end_year),
            min_rating=min_rating,
        )
        results = results_payload(frame)
        if not results:
            suggestions = recommender.get_closest_titles(query, n=5)

    return render_template(
        "similar.html",
        query=query,
        top_n=top_n,
        min_similarity=min_similarity,
        start_year=start_year,
        end_year=end_year,
        min_rating=min_rating,
        min_year=min_year,
        max_year=max_year,
        results=results,
        suggestions=suggestions,
    )


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommender = get_recommender()
    min_year, max_year = year_bounds(recommender)
    liked = request.values.get("liked", "").strip()
    liked_titles = [title.strip() for title in liked.splitlines() if title.strip()]
    top_n = parse_int("top_n", 10)
    min_similarity = parse_float("min_similarity", 0.1)
    start_year = parse_int("start_year", min_year)
    end_year = parse_int("end_year", max_year)
    min_rating = parse_float("min_rating", 0.0)
    results = []
    suggestions = {}

    if liked_titles:
        frame = recommender.recommend_for_user(
            liked_titles,
            top_n=top_n,
            min_similarity=min_similarity,
            year_range=(start_year, end_year),
            min_rating=min_rating,
        )
        results = results_payload(frame)
        if not results:
            suggestions = {title: recommender.get_closest_titles(title, n=3) for title in liked_titles}

    return render_template(
        "recommend.html",
        liked=liked,
        top_n=top_n,
        min_similarity=min_similarity,
        start_year=start_year,
        end_year=end_year,
        min_rating=min_rating,
        min_year=min_year,
        max_year=max_year,
        results=results,
        suggestions=suggestions,
    )


@app.route("/search")
def search():
    recommender = get_recommender()
    query = request.args.get("query", "").strip()
    limit = parse_int("limit", 20)
    results = []
    if query:
        results = results_payload(recommender.search_movies(query, limit=limit))
    return render_template("search.html", query=query, limit=limit, results=results)


@app.route("/analytics")
def analytics():
    recommender = get_recommender()
    stats = recommender.get_statistics()
    year_distribution = recommender.get_year_distribution()
    top_years = sorted(year_distribution.items(), key=lambda item: item[1], reverse=True)[:15]
    top_keywords = list(recommender.get_top_keywords(25).items())
    return render_template("analytics.html", stats=stats, top_years=top_years, top_keywords=top_keywords)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/download/<kind>")
def download(kind):
    recommender = get_recommender()
    if kind == "similar":
        query = request.args.get("query", "").strip()
        if not query:
            return Response("Missing query", status=400)
        frame = recommender.recommend_similar_movies(
            query,
            top_n=parse_int("top_n", 20),
            min_similarity=parse_float("min_similarity", 0.1),
            year_range=(parse_int("start_year", 0), parse_int("end_year", 9999)),
            min_rating=parse_float("min_rating", 0.0),
        )
        filename = "similar_movies.csv"
    elif kind == "recommend":
        liked = request.args.get("liked", "").strip()
        liked_titles = [title.strip() for title in liked.splitlines() if title.strip()]
        if not liked_titles:
            return Response("Missing liked movies", status=400)
        frame = recommender.recommend_for_user(
            liked_titles,
            top_n=parse_int("top_n", 30),
            min_similarity=parse_float("min_similarity", 0.1),
            year_range=(parse_int("start_year", 0), parse_int("end_year", 9999)),
            min_rating=parse_float("min_rating", 0.0),
        )
        filename = "recommendations.csv"
    else:
        return Response("Unknown download type", status=404)

    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    get_recommender()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
