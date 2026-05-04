# Movie Recommender System - Flask Web App

A polished Flask web app for discovering movies you'll love.

## Quick Start

### Easiest Windows Run
```powershell
.\run_app.ps1
```

Or double-click `run_app.bat`.

If `.venv` or the MovieLens data is missing, the run script will call `setup_project.ps1` first.

### Fresh Laptop Setup
After formatting/reinstalling Windows, install Python 3.11+ first, then run:

```powershell
.\setup_project.ps1
.\run_app.ps1
```

The setup script creates `.venv`, installs dependencies, downloads the `ml-32m` dataset if it is missing, checks packages, and runs tests.

### Manual Install
```powershell
pip install -r requirements.txt
```

### Manual Run
```powershell
python app.py
```

The app will open in your browser at `http://localhost:5000`.

## Features

- Home page with dataset summary
- Similar movie search
- Personalized recommendations from liked movies
- Movie title search
- Analytics for years and keywords
- CSV download for recommendation results
- Flask templates and custom CSS frontend

## How It Works

1. Movies are loaded and preprocessed on startup.
2. Titles, genres, and tags are combined into text features.
3. TF-IDF vectors and cosine similarity find related movies.
4. Rating and popularity boosts improve result quality.
5. Flask routes render the results as normal web pages.

## Files in This Project

| File | Purpose |
|------|---------|
| `app.py` | Flask web application |
| `templates/` | HTML templates |
| `static/styles.css` | App styling |
| `movie_complete.py` | Core recommendation engine |
| `movie_optimized.py` | Optimized version |
| `title_based_recommender_script.py` | Standalone script |
| `ml-32m/movies.csv` | Movie dataset |
| `requirements.txt` | Python dependencies |

## Pages

1. Home - Overview and quick insights
2. Find Similar - Search for similar movies
3. Personal Picks - Get personalized suggestions
4. Search - Full-text movie search
5. Analytics - Detailed statistics
6. About - System information

## Command Reference

```powershell
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# Run tests
python -m pytest -q
```

## Deployment

This Flask app can be deployed on services like Render, Railway, Heroku, or a VPS. For production, use a WSGI server such as Gunicorn on Linux.

**Made with Flask**
