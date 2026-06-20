import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("About")

st.sidebar.success("AI/ML Project by Hema")

st.sidebar.info("""
Movie Recommendation System

Built using:
- Python
- Pandas
- Scikit-Learn
- Streamlit
- Cosine Similarity
""")

st.sidebar.write("Dataset: MovieLens")

# ==========================================
# Load Dataset
# ==========================================

movies = pd.read_csv("data/movies.csv")

# ==========================================
# Data Preprocessing
# ==========================================

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# ==========================================
# Feature Extraction
# ==========================================

cv = CountVectorizer()

genre_matrix = cv.fit_transform(movies['genres'])

# ==========================================
# Similarity Matrix
# ==========================================

similarity = cosine_similarity(genre_matrix)

# ==========================================
# Recommendation Function
# ==========================================

def recommend(movie_name):

    movie_index = movies[movies['title'] == movie_name].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for movie in movie_list:

        recommendations.append({
            "title": movies.iloc[movie[0]].title,
            "genre": movies.iloc[movie[0]].genres,
            "score": round(movie[1] * 100, 2)
        })

    return recommendations

# ==========================================
# Main UI
# ==========================================

st.title("🎬 Movie Recommendation System")

st.markdown("""
Discover movies similar to your favorites using
Machine Learning and Content-Based Filtering.
""")

# ==========================================
# Project Statistics
# ==========================================

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.metric("Movies", len(movies))

with stat2:
    st.metric("Unique Genre Combinations",
              movies['genres'].nunique())

with stat3:
    st.metric("Algorithm", "Cosine Similarity")

st.divider()

# ==========================================
# Movie Selection
# ==========================================

col1, col2 = st.columns([3, 1])

with col1:
    movie_name = st.selectbox(
        "🎥 Search and Select a Movie",
        movies['title'].values
    )

with col2:
    st.metric(
        "Movies Available",
        len(movies)
    )

# ==========================================
# Recommendation Button
# ==========================================

if st.button("🎬 Recommend"):

    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):

        st.write(f"### {i}. {movie['title']}")

        st.caption(movie['genre'])

        st.progress(movie['score'] / 100)

        st.write(f"Similarity Score: {movie['score']}%")

        st.write("")

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown(
    """
    Built by **Hema** using **Python, Scikit-Learn and Streamlit** 🚀
    """
)