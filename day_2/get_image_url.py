import requests

TMDB_API_KEY = "aed0f7f5a89450f11167d5e0254cff70"
BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def get_poster_url(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    poster_path = data.get("poster_path")
    if not poster_path:
        return None

    return f"{BASE_IMAGE_URL}{poster_path}"
