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
tags["tag"] = tags["tag"].fillna("")

# Combine all user tags for each movie
movie_tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x)
).reset_index()

# Merge tags with movie data
movies = movies.merge(movie_tags, on="movieId", how="left")

movies["tag"] = movies["tag"].fillna("")

# Combine title, genres and tags
movies["features"] = (
    movies["title"].str.replace(r"\(\d{4}\)", "", regex=True)
    + " "
    + movies["genres"].str.replace("|", " ", regex=False)
    + " "
    + movies["tag"]
)


# ==========================================
# TF-IDF Feature Extraction
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["features"])


# ==========================================
# Cosine Similarity
# ==========================================

similarity_matrix = cosine_similarity(tfidf_matrix)
# ==========================================
# Recommendation Function
# ==========================================

def recommend_movies(movie_name, top_n=10):

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
