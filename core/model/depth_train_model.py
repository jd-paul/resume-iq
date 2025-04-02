"""
NOTES: Understanding the Depth Model (Logistic Regression + Sentence Embeddings)
--------------------------------------------------------------------------------

### What This Model Does**
This model predicts whether a resume bullet point demonstrates strong "depth":
  - Clear, specific action (e.g., "Built", "Engineered", "Automated")
  - Technical or methodological detail (tools, techniques, context)
  - Contextual impact (metrics, platform, scale)

It uses a pre-trained **sentence-transformer embedding model** to convert text into numerical vectors,
then applies a **Logistic Regression classifier** to predict binary depth labels:
    1 = Deep bullet point
    0 = Shallow or vague

### Step-by-Step Breakdown**
#### Data Preparation**
- Labeled data is stored in `depth_data.txt` in the format: `sentence | label`
- `load_data()`:
    - Reads and cleans each line
    - Extracts sentences and their corresponding labels (as integers)

- The dataset `depth_data.txt` contains labeled sentences (`sentence | label` format).
- `load_data()`:
    - Reads and cleans each line
    - Extracts sentences and their corresponding labels (as integers)

#### Feature Extraction (TF-IDF)**
- Uses `sentence-transformers/all-mpnet-base-v2` to convert each sentence into a 768-dimensional vector
- These embeddings capture contextual meaning better than bag-of-words or TF-IDF
- Example: "Built a backend in Node" vs "Handled backend work" → embeddings capture the nuance

#### Model Training (Logistic Regression)**
- Trains a binary classifier on top of sentence embeddings
- **Logistic Regression** is used because:
    - It is lightweight and interpretable
    - Outputs a probability score (0 to 1) for each sentence
- Input: `X_train` = embeddings, `y_train` = labels

D. Model Evaluation
- The data is split into 80% training and 20% test
- `classification_report()` is printed to evaluate performance (precision, recall, f1-score)

E. Saving the Model
- The trained classifier is saved to `depth_model.pkl`
- Can be loaded later using `depth_model.py` wrapper for prediction

Example Inference Flow
-----------------------
- A sentence is embedded → passed to the classifier → outputs probability:
    "Built a CI/CD pipeline..." → 0.92 (Deep)
    "Worked on projects..."     → 0.21 (Shallow)

"""


import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/depth_data.txt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'depth_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'depth_vectorizer.pkl')

def load_data():
    texts, labels = [], []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            if "|" not in line:
                continue
            sentence, label = line.rsplit("|", 1)
            sentence = sentence.strip()
            label = label.strip()
            if label in {"0", "1"}:
                texts.append(sentence)
                labels.append(int(label))
    return texts, labels

def train_model():
    # Load data
    X, y = load_data()
    print(f"Loaded {len(X)} examples.")

    # Encode text using sentence-transformers
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    # Train simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
