# Movie Recommender

Content-based movie recommender (TMDB 5000 dataset) with a Streamlit UI.

## Setup
```bash
pip install -r requirements.txt
```
1. Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` (Kaggle: "TMDB Movie Metadata") into `./data`.
2. Build the model: `python build_model.py`
3. Get a free API key at https://www.themoviedb.org/settings/api, then copy
   `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste the key in.
4. Run: `streamlit run app.py`

## How it works
Each movie becomes a text "tags" document (plot + genres + keywords + cast + director),
stemmed and vectorised with TF-IDF. Similarity = cosine similarity between movies.
The final score optionally blends in a weighted rating so good movies rank higher.
