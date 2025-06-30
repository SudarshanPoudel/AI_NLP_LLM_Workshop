"""
This is a minimal code, which is then sent to chatgpt to convert into streamlit ui (app.py)
"""

import pandas as pd
import joblib
from get_image_url import get_image_url

df = pd.read_csv("../dataset/movies.csv")
df.head()

def combine_fields(row):
    return (
        f"{row['title']} directed by {row['director']}. "
        f"Genre: {row['genres']}. "
        f"Starring: {row['actors']}. "
        f"{row['description']}"
    )

df['full_text'] = df.apply(combine_fields, axis=1)

embedding_model = joblib.load("embedding_model.pkl")
knn = joblib.load("knn.pkl")

def recommend_movies(index, no_of_movies=5):
    movie_row = df.iloc[index]
    print(f"Selected Movie: {movie_row['title']}")
    query_vector = embedding_model.encode([movie_row['full_text']])
    distances, indices = knn.kneighbors(query_vector, n_neighbors=no_of_movies+1)  
    recommended_df = df.iloc[indices[0][1:]]  

    return recommended_df

def recommend_movies_by_text(text, no_of_movies=5):
    query_vector = embedding_model.encode([text])
    distances, indices = knn.kneighbors(query_vector, n_neighbors=no_of_movies)  
    recommended_df = df.iloc[indices[0]]  

    return recommended_df