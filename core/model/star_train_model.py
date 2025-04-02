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

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from core.model.star_load_data import load_data, preprocess_data
from sklearn.model_selection import train_test_split, GridSearchCV

def train_model(data):
    """
    Trains a Logistic Regression model on the labeled data.
    """
    X = data["text"]
    y = data["label"]

    # Create TF-IDF vectorizer with n-grams
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)

    # Hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2']
    }
    lr = LogisticRegression()
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_vectorized, y)

    best_model = grid_search.best_estimator_

    # Save best model & vectorizer
    joblib.dump(best_model, "core/model/star_model.pkl")
    joblib.dump(vectorizer, "core/model/star_vectorizer.pkl")

    # Evaluate with cross_val_score or a train_test_split again
    y_pred = best_model.predict(X_vectorized)
    accuracy = (y_pred == y).mean() * 100
    print(f"Best model accuracy on entire dataset: {accuracy:.2f}%")


if __name__ == "__main__":
    # Load and preprocess data
    data = load_data("core/data/star_data.txt")
    data = preprocess_data(data)

    # Train the model
    train_model(data)


