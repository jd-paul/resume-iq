"""
**Notes: Understanding the STAR Model (Logistic Regression + TF-IDF)**
----------------------------------------------------------------------

### What This Model Does**
- This model predicts whether a given sentence follows the **STAR (Situation, Task, Action, Result) method**.
- It uses **Logistic Regression**, a classification algorithm, trained on labeled sentences (`1` = STAR, `0` = Not STAR).
- Input text is **converted into numerical features** using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

---

### Step-by-Step Breakdown**
#### Data Preparation**
- The dataset `star_data.txt` contains labeled sentences (`sentence | label` format).
- `load_data()` reads this file, and `preprocess_data()`:
  - Converts labels (`1` or `0`) to integers.
  - Drops any missing values.

#### Feature Extraction (TF-IDF)**
- Textual data is converted into a numerical format using **TF-IDF**:
  - **TF (Term Frequency):** How often a word appears in a document.
  - **IDF (Inverse Document Frequency):** Reduces the importance of words that appear frequently across all documents.
  - The result is a **sparse matrix representation** of sentences.

#### Model Training (Logistic Regression)**
- **Why Logistic Regression?**
  - A simple yet powerful **binary classification** model.
  - Outputs a probability score between **0 and 1** for STAR classification.
- The model is trained on:
  - **X_train (TF-IDF sentence vectors)**
  - **y_train (STAR labels: 1 or 0)**

#### Model Evaluation**
- The dataset is split into **80% training / 20% testing
- Accuracy is calculated on the test set:
  ```python
  accuracy = model.score(X_test_vectorized, y_test)

"""

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def load_data(file_path: str):
    """Loads labeled STAR data from a 'sentence | label' text file."""
    return pd.read_csv(file_path, sep="|", header=None, names=["text", "label"])

def preprocess_data(data: pd.DataFrame):
    """Cleans the dataset by converting labels to int and dropping incomplete rows."""
    data["label"] = data["label"].astype(int)
    return data.dropna()

def train_model(data_path: str):
    """
    Loads data, vectorizes text using TF-IDF, trains a Logistic Regression classifier,
    and saves both the model and vectorizer to disk.
    """
    data = preprocess_data(load_data(data_path))
    X, y = data["text"], data["label"]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    grid = GridSearchCV(LogisticRegression(), {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"]
    }, cv=5, scoring="f1")

    grid.fit(X_vec, y)
    best_model = grid.best_estimator_

    joblib.dump(best_model, "core/model/star_model.pkl")
    joblib.dump(vectorizer, "core/model/star_vectorizer.pkl")

    print(f"✔ Trained STAR model — Accuracy: {(best_model.predict(X_vec) == y).mean():.2%}")

if __name__ == "__main__":
    train_model("core/data/star_data.txt")
