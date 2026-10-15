import requests

KEY = "daff46a68412fc5f26f205fd28fedfda"
r = requests.get("https://api.themoviedb.org/3/movie/19995", params={"api_key": KEY}, timeout=10)
print(r.status_code)
print(r.text[:300])