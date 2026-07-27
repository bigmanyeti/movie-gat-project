"""
Downloads and prepares the REAL TMDB 5000 Movie Dataset — no synthetic
or randomly assigned metadata. Director, cast (actors), and
vote_average (real audience rating) are all genuine data pulled from
The Movie Database (TMDB), sourced via the well-known "tmdb_5000"
movies+credits CSV pair (originally distributed on Kaggle; mirrored
here from a public GitHub raw file so no API key is required).

Dataset contents (real, not synthesized):
    - 4,803 movies
    - Real genres, real release year
    - Real vote_average / vote_count (TMDB's aggregated user rating)
    - Real cast (top-billed actors) and real crew (Director) pulled
      from TMDB's structured JSON crew field
      (producer is intentionally not extracted -- removed project-wide)

Run this once before running the Streamlit app:
    python dataset/download_data.py
"""

import os
import ast
import urllib.request

import pandas as pd

MOVIES_URL = "https://raw.githubusercontent.com/andandandand/CSV-datasets/master/tmdb_5000_movies.csv"
CREDITS_URL = "https://raw.githubusercontent.com/andandandand/CSV-datasets/master/tmdb_5000_credits.csv"

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

MAX_CAST_PER_MOVIE = 5  # top-billed actors only (credits.json is ordered by billing)


def _download_file(url, dest_path, label):
    if not os.path.exists(dest_path):
        print(f"Downloading {label}...")
        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def _download_tmdb():
    os.makedirs(RAW_DIR, exist_ok=True)
    movies_path = _download_file(MOVIES_URL, os.path.join(RAW_DIR, "tmdb_5000_movies.csv"), "TMDB movies")
    credits_path = _download_file(CREDITS_URL, os.path.join(RAW_DIR, "tmdb_5000_credits.csv"), "TMDB credits")
    return movies_path, credits_path


def _safe_literal_eval(value):
    """TMDB's genres/cast/crew columns are JSON-like strings (Python
    literal lists of dicts). Parses safely, returning [] on failure
    instead of crashing on the rare malformed row."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def _extract_genres(genres_raw):
    parsed = _safe_literal_eval(genres_raw)
    return "|".join(g["name"] for g in parsed) if parsed else ""


def _extract_cast(cast_raw, top_n=MAX_CAST_PER_MOVIE):
    parsed = _safe_literal_eval(cast_raw)
    parsed_sorted = sorted(parsed, key=lambda c: c.get("order", 999))
    return "|".join(c["name"] for c in parsed_sorted[:top_n])


def _extract_director(crew_raw):
    parsed = _safe_literal_eval(crew_raw)
    directors = [c["name"] for c in parsed if c.get("job") == "Director"]
    return directors[0] if directors else "Unknown Director"


def _extract_year(release_date):
    if isinstance(release_date, str) and len(release_date) >= 4:
        try:
            return int(release_date[:4])
        except ValueError:
            return None
    return None


def build_dataset(force_rebuild=False):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "movies_enriched.csv")

    if not force_rebuild and os.path.exists(out_path):
        print("Processed dataset already exists.")
        return pd.read_csv(out_path)

    movies_path, credits_path = _download_tmdb()

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # credits.csv has its own "title" + "movie_id"; join on title since
    # movies.csv's "id" corresponds to credits.csv's "movie_id".
    credits = credits.rename(columns={"movie_id": "id"})
    merged = movies.merge(credits[["id", "cast", "crew"]], on="id", how="inner")

    merged["genres"] = merged["genres"].apply(_extract_genres)
    merged["actors"] = merged["cast"].apply(_extract_cast)
    merged["director"] = merged["crew"].apply(_extract_director)
    merged["year"] = merged["release_date"].apply(_extract_year)
    merged["year"] = merged["year"].fillna(merged["year"].median()).astype(int)

    # Drop movies with no genre or no cast info — not enough signal to
    # place them meaningfully in the graph.
    merged = merged[(merged["genres"] != "") & (merged["actors"] != "")]

    # Rename to match the rest of the pipeline's expected schema
    # (movieId, title, genres, director, actors, year, avg_rating,
    # num_ratings) — avg_rating/num_ratings now come directly from
    # TMDB's own real aggregated user ratings, not MovieLens ratings.
    # NOTE: producer is intentionally NOT extracted/kept -- it has been
    # removed entirely from this project (graph nodes/edges, the GAT
    # feature vector, and the traditional baseline's weights).
    out = merged.rename(columns={
        "id": "movieId",
        "vote_average": "avg_rating",
        "vote_count": "num_ratings",
    })[[
        "movieId", "title", "genres", "director", "actors",
        "year", "avg_rating", "num_ratings",
    ]]

    out = out.drop_duplicates(subset="movieId").reset_index(drop=True)

    out.to_csv(out_path, index=False)
    print(f"Saved enriched dataset to {out_path}")
    print(f"Movies: {len(out)} (real TMDB data — no synthetic fields)")
    return out


if __name__ == "__main__":
    df = build_dataset(force_rebuild=True)
    print(df.head())
    print(f"Total movies: {len(df)}")