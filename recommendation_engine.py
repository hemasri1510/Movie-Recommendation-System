import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Load Data
# ==========================================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")


# ==========================================
# Prepare Movie Data
# ==========================================

movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)


# ==========================================
# Prepare Tags
# ==========================================

tags["tag"] = tags["tag"].fillna("")

movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

movies = movies.merge(movie_tags, on="movieId", how="left")

movies["tag"] = movies["tag"].fillna("")


# ==========================================
# Create Content Features
# ==========================================

movies["content"] = (
    movies["genres"] + " " + movies["tag"]
)


# ==========================================
# TF-IDF Feature Extraction
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["content"])


# ==========================================
# Cosine Similarity
# ==========================================

similarity_matrix = cosine_similarity(tfidf_matrix)


# ==========================================
# Recommendation Function
# ==========================================

def recommend_movies(movie_name, top_n=5):

    movie_index = movies[
        movies["title"] == movie_name
    ].index[0]

    distances = similarity_matrix[movie_index]

    movie_indices = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:top_n + 1]

    recommendations = []

    for index, score in movie_indices:

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genre": movies.iloc[index]["genres"],
            "similarity": round(score, 3)
        })

    return recommendations
