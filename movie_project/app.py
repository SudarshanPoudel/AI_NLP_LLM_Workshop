import streamlit as st
import pandas as pd
import joblib
from get_image_url import get_image_url

# Load data and models
df = pd.read_csv("../dataset/movies.csv")
embedding_model = joblib.load("embedding_model.pkl")
knn = joblib.load("knn.pkl")

# Prepare full_text column for embedding
def combine_fields(row):
    return (
        f"{row['title']} directed by {row['director']}. "
        f"Genre: {row['genres']}. "
        f"Starring: {row['actors']}. "
        f"{row['description']}"
    )
df['full_text'] = df.apply(combine_fields, axis=1)

# Recommendation function
def recommend_movies(index, no_of_movies=10):
    query_vector = embedding_model.encode([df.iloc[index]['full_text']])
    _, indices = knn.kneighbors(query_vector, n_neighbors=no_of_movies + 1)
    return df.iloc[indices[0][1:]]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# Movie selection dropdown
movie_titles = df['title'].tolist()
selected_movie_title = st.selectbox("Search for a movie", sorted(movie_titles))

# Show selected movie details
selected_index = df[df['title'] == selected_movie_title].index[0]
selected_movie = df.iloc[selected_index]

st.markdown("### 🎥 Selected Movie")
col1, col2 = st.columns([1, 2])
with col1:
    image_url = get_image_url(selected_movie['id'])
    st.image(image_url, use_container_width=True)
with col2:
    st.markdown(f"**Title:** {selected_movie['title']}")
    st.markdown(f"**Director:** {selected_movie['director']}")
    st.markdown(f"**Genres:** {selected_movie['genres']}")
    st.markdown(f"**Actors:** {selected_movie['actors']}")
    st.markdown(f"**Release Date:** {selected_movie['release_date']}")
    st.markdown(f"**Rating:** {selected_movie['rating']}")
    st.markdown(f"**Vote Count:** {selected_movie['vote_count']}")
    st.markdown(f"**Description:** {selected_movie['description']}")

# Recommended movies
st.markdown("### 🔍 Recommended Movies")
recommended = recommend_movies(selected_index)

cols = st.columns(5)
for i, (idx, row) in enumerate(recommended.iterrows()):
    with cols[i % 5]:
        img_url = get_image_url(row['id'])
        st.image(img_url, use_container_width=True, caption=row['title'])
