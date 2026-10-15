"""
Build the recommender model from the TMDB 5000 dataset.

Usage:
    1. Put tmdb_5000_movies.csv and tmdb_5000_credits.csv in ./data
    2. python build_model.py

Outputs (in ./model):
    movies.pkl  - cleaned movie metadata (DataFrame)
    tfidf.npz   - sparse TF-IDF matrix (one row per movie)
"""
import ast
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

stemmer = PorterStemmer()


def parse_names(text, limit=None):
    """Turn a JSON-like string of dicts into a list of 'name' values."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    names = [i["name"] for i in items]
    return names[:limit] if limit else names


def parse_director(text):
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [i["name"] for i in items if i.get("job") == "Director"][:1]


def squash(names):
    """'Tom Hanks' -> 'tomhanks' so first names don't match across people."""
    return [n.replace(" ", "").lower() for n in names]


def stem_text(text):
    return " ".join(stemmer.stem(w) for w in text.split())


def main():
    movies = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")

    # Merge on ID (not title) so remakes / same-title movies don't get duplicated
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"])
    df = movies.merge(credits, on="id")

    df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew",
             "release_date", "vote_average", "vote_count", "popularity"]]
    df = df.dropna(subset=["overview"]).drop_duplicates(subset="title").reset_index(drop=True)
    df = df.rename(columns={"id": "movie_id"})

    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["cast"] = df["cast"].apply(lambda t: parse_names(t, limit=4))
    df["director"] = df["crew"].apply(parse_director)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(0).astype(int)

    # Build the "tags" text. Repeating a field = giving it more weight.
    def make_tags(row):
        parts = (
            row["overview"].lower().split()
            + squash(row["genres"]) * 3        # genre matters a lot
            + squash(row["keywords"])
            + squash(row["cast"])
            + squash(row["director"]) * 2      # same director = strong signal
        )
        return stem_text(" ".join(parts))

    df["tags"] = df.apply(make_tags, axis=1)

    tfidf = TfidfVectorizer(max_features=8000, stop_words="english",
                            ngram_range=(1, 2), min_df=2)
    matrix = tfidf.fit_transform(df["tags"])  # L2-normalised -> dot product == cosine

    keep = ["movie_id", "title", "year", "overview", "genres", "cast",
            "director", "vote_average", "vote_count", "popularity"]
    with open(MODEL_DIR / "movies.pkl", "wb") as f:
        pickle.dump(df[keep], f)
    sp.save_npz(MODEL_DIR / "tfidf.npz", matrix)

    print(f"Done: {len(df)} movies, matrix shape {matrix.shape}")


if __name__ == "__main__":
    main()
