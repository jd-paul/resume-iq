"""
📌 **Notes: Understanding the STAR Model (Logistic Regression + TF-IDF)**
==============================================================

### What This Model Does**
- This model predicts whether a given sentence follows the **STAR (Situation, Task, Action, Result) method**.
- It uses **Logistic Regression**, a classification algorithm, trained on labeled sentences (`1` = STAR, `0` = Not STAR).
- Input text is **converted into numerical features** using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

---

### Step-by-Step Breakdown**
#### **A. Data Preparation**
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
from core.model.load_data import load_data, preprocess_data

def train_model(data):
    """
    Trains a Logistic Regression model on the labeled data.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Save the model and vectorizer to the correct path
    joblib.dump(model, "core/model/star_model.pkl")  # ✅ Fixed model save path
    joblib.dump(vectorizer, "core/model/star_vectorizer.pkl")  # ✅ Fixed vectorizer save path

    # Evaluate the model
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data("core/data/star_data.txt")
    data = preprocess_data(data)

    # Train the model
    train_model(data)
