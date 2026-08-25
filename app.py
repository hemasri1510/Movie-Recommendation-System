iimport streamlit as st

from recommendation_engine import movies, recommend_movies

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
- TF-IDF
- Cosine Similarity
- Streamlit
""")

st.sidebar.write("Dataset: MovieLens")

# ==========================================
# Main UI
# ==========================================

st.title("🎬 Movie Recommendation System")

st.markdown("""
Discover movies similar to your favorites using
TF-IDF, movie genres, user tags, and content-based filtering.
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
    st.metric("Model", "TF-IDF + Cosine")

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

    recommendations = recommend_movies(movie_name)

    st.subheader("Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):

        st.write(f"### {i}. {movie['title']}")

        st.caption(movie["genre"])

        score = movie["similarity"]

        st.progress(score)

        st.write(f"Similarity Score: {score:.3f}")

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
