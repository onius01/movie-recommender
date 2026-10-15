import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import scipy.sparse as sp
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same folder as app.py, wherever you launch from
load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

PLACEHOLDER = "https://placehold.co/500x750?text=No+Poster"


# ---------- data loading (cached so it only runs once) ----------
@st.cache_resource
def load_model():
    with open("model/movies.pkl", "rb") as f:
        movies = pickle.load(f)
    matrix = sp.load_npz("model/tfidf.npz").tocsr()

    # IMDb-style weighted rating: stops a 10/10 movie with 3 votes from winning
    C = movies["vote_average"].mean()
    m = movies["vote_count"].quantile(0.60)
    v, R = movies["vote_count"], movies["vote_average"]
    movies["quality"] = ((v / (v + m)) * R + (m / (v + m)) * C) / 10  # 0..1
    return movies, matrix


def get_api_key():
    return os.environ.get("TMDB_API_KEY")


def fetch_poster(movie_id, api_key):
    if not api_key:
        return PLACEHOLDER
    return _fetch_poster_cached(movie_id, api_key)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _fetch_poster_cached(movie_id, api_key):
    r = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": api_key},
        timeout=10,
    )
    r.raise_for_status()                      # errors are NOT cached
    path = r.json().get("poster_path")
    return f"https://image.tmdb.org/t/p/w500{path}" if path else PLACEHOLDER


# ---------- recommendation logic ----------
def recommend(title, movies, matrix, n=5, quality_weight=0.2, genre_filter=None):
    idx = movies.index[movies["title"] == title][0]
    sims = (matrix @ matrix[idx].T).toarray().ravel()   # cosine similarity (rows are L2-normalised)
    score = (1 - quality_weight) * sims + quality_weight * movies["quality"].values
    score[idx] = -1                                      # never recommend the movie itself

    if genre_filter:
        mask = movies["genres"].apply(lambda g: bool(set(g) & set(genre_filter))).values
        score = np.where(mask, score, -1)

    top = np.argsort(score)[::-1][:n]
    result = movies.iloc[top].copy()
    result["similarity"] = sims[top]
    return result


# ---------- UI ----------
st.title("🎬 Movie Recommender")
st.caption("Content-based: finds movies with similar plot, genre, cast, director and keywords.")

try:
    movies, matrix = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run `python build_model.py` first.")
    st.stop()

api_key = get_api_key()
if not api_key:
    st.warning("No TMDB_API_KEY set, so posters are disabled. See README.")

with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 5, 15, 5)
    quality_weight = st.slider(
        "Boost well-rated movies", 0.0, 0.6, 0.2, 0.05,
        help="0 = purely similar. Higher = prefer popular, highly rated movies.",
    )
    all_genres = sorted({g for gs in movies["genres"] for g in gs})
    genre_filter = st.multiselect("Only show these genres (optional)", all_genres)

selected = st.selectbox(
    "Pick a movie you like",
    movies["title"].values,
    index=None,
    placeholder="Type to search...",
)

if selected:
    row = movies[movies["title"] == selected].iloc[0]
    with st.expander(f"About {selected} ({row['year'] or 'n/a'})", expanded=False):
        st.write(row["overview"])
        st.write(f"**Director:** {', '.join(row['director']) or 'n/a'}")
        st.write(f"**Cast:** {', '.join(row['cast'])}")
        st.write(f"**Genres:** {', '.join(row['genres'])}")

if st.button("Show Recommendations", type="primary", disabled=not selected):
    recs = recommend(selected, movies, matrix, n, quality_weight, genre_filter)

    if recs.empty or (recs["similarity"] <= 0).all():
        st.info("No good matches found. Try removing the genre filter.")
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:   # fetch posters in parallel
            posters = list(pool.map(lambda i: fetch_poster(i, api_key), recs["movie_id"]))

        for start in range(0, len(recs), 5):              # rows of 5
            cols = st.columns(5)
            for col, (_, r), poster in zip(
                cols, recs.iloc[start:start + 5].iterrows(), posters[start:start + 5]
            ):
                with col:
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**{r['title']}** ({r['year'] or 'n/a'})")
                    st.caption(f"⭐ {r['vote_average']:.1f}  •  {r['similarity']:.0%} match")
                    with st.expander("Plot"):
                        st.write(r["overview"])

st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")