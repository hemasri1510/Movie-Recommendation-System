# 🎬 Movie Recommendation System

An interactive **machine learning-based movie recommendation system** that recommends movies similar to a selected movie using **content-based filtering and cosine similarity**.

Built with **Python, Pandas, Scikit-learn and Streamlit**.

---

## 📌 Overview

Finding a movie to watch can be difficult when there are thousands of options available.

This project uses the **genres of movies** to identify similarities between them and recommend movies that have similar genre profiles.

The recommendation engine converts movie genres into numerical feature vectors and calculates the similarity between movies using **cosine similarity**.

---

## ✨ Features

* 🎥 Select a movie from the available dataset
* 🤖 Content-based movie recommendations
* 🎯 Recommends the **Top 5 similar movies**
* 📊 Displays similarity scores
* 🎭 Shows the genres of recommended movies
* 📈 Displays basic dataset statistics
* 🖥️ Interactive Streamlit interface

---

## 🧠 Machine Learning Approach

The recommendation pipeline works in the following steps:

```text
Movie Dataset
      ↓
Genre Preprocessing
      ↓
CountVectorizer
      ↓
Genre Feature Matrix
      ↓
Cosine Similarity
      ↓
Similarity Ranking
      ↓
Top 5 Recommendations
```

### 1. Data Preprocessing

Movie genres are originally represented using the `|` separator.

For example:

```text
Action|Adventure|Sci-Fi
```

The separator is replaced with spaces before feature extraction.

### 2. Feature Extraction

`CountVectorizer` from Scikit-learn converts the movie genres into numerical feature vectors.

### 3. Similarity Calculation

**Cosine Similarity** is used to measure how similar two movies are based on their genre vectors.

### 4. Recommendation

For the selected movie, the system ranks other movies according to their similarity scores and returns the **top 5 recommendations**.

---

## 🛠️ Tech Stack

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Application development   |
| Pandas            | Data processing           |
| Scikit-learn      | Machine learning          |
| CountVectorizer   | Feature extraction        |
| Cosine Similarity | Similarity calculation    |
| Streamlit         | Interactive web interface |
| Jupyter Notebook  | Data exploration          |

---

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── data/
│   └── movies.csv
│
├── app.py
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/hemasri1510/Movie-Recommendation-System.git
```

### 2. Navigate to the project

```bash
cd Movie-Recommendation-System
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Example

Select a movie from the application and click **Recommend**.

The system returns:

```text
1. Recommended Movie
   Genre: ...
   Similarity Score: ...%

2. Recommended Movie
   Genre: ...
   Similarity Score: ...%

3. Recommended Movie
   Genre: ...
   Similarity Score: ...%
```

---

## 🔮 Future Improvements

This project is currently a **content-based recommendation system using movie genres**.

Planned improvements include:

* [ ] Replace CountVectorizer with **TF-IDF**
* [ ] Use additional movie metadata such as titles and tags
* [ ] Build a **hybrid recommendation system**
* [ ] Add movie posters and descriptions
* [ ] Improve the user interface
* [ ] Add recommendation evaluation metrics
* [ ] Deploy the application
* [ ] Add more personalized recommendations

---

## 👩‍💻 Author

**Hema Sri**

AI & ML enthusiast interested in building practical solutions using technology and machine learning.

---

⭐ If you find this project interesting, feel free to explore the repository!
