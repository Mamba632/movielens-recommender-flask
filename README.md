# MovieLens Recommender Flask

A Flask web application for discovering similar movies and generating personalized recommendations from the MovieLens 32M dataset.

## Features

- Similar movie recommendations using TF-IDF and cosine similarity
- Personalized recommendations from selected liked movies
- Movie title search
- Dataset analytics for years, genres, and keywords
- CSV export for recommendation results
- Responsive Flask frontend with custom templates and CSS
- Automated tests for the recommender logic

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- SciPy
- pytest
- Git LFS for large MovieLens dataset files

## Dataset

This project uses the MovieLens 32M dataset in the `ml-32m/` folder.

Large dataset files are stored with Git LFS:

- `ml-32m/ratings.csv`
- `ml-32m/tags.csv`

## Clone And Setup

Install Git LFS before cloning or pulling the dataset files:

```bash
git lfs install
git clone https://github.com/Mamba632/movielens-recommender-flask.git
cd movielens-recommender-flask
git lfs pull
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

On Windows, you can also run:

```powershell
.\run_app.ps1
```

## Run Tests

```bash
python -m pytest -q
```

## Project Structure

```text
movielens-recommender-flask/
|-- app.py
|-- movie_complete.py
|-- movie_optimized.py
|-- title_based_recommender.py
|-- templates/
|-- static/
|-- tests/
|-- ml-32m/
|-- requirements.txt
|-- README_DASHBOARD.md
`-- README.md
```

## Notes

- First startup can take some time because the app loads the MovieLens data and prepares recommendation features.
- If `ratings.csv` or `tags.csv` look very small after cloning, run `git lfs pull`.
- `README_DASHBOARD.md` contains the original dashboard-focused project notes.

## Author

Manav Raval
