"""
depth_model.py

Wrapper for the trained depth classifier model (depth_model.pkl) using sentence-transformers.

This allows the rest of the Resume IQ system to easily evaluate bullet points for technical and methodological depth.
"""

import os
import joblib
from sentence_transformers import SentenceTransformer

class DepthModel:
    def __init__(self, model_path="core/model/depth_model.pkl"):
        """
        Loads the sentence-transformer model and the trained classifier.
        """
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.classifier = joblib.load(model_path)

    def predict_proba(self, bullet: str) -> float:
        """
        Returns probability that a bullet is 'deep' (1.0 = high depth, 0.0 = low depth).
        """
        embedding = self.embedder.encode([bullet])
        return self.classifier.predict_proba(embedding)[0][1]  # probability of class 1

    def is_deep(self, bullet: str, threshold=0.5) -> bool:
        """
        Returns True if bullet is deep enough (above threshold).
        """
        return self.predict_proba(bullet) >= threshold

if __name__ == "__main__":
    dm = DepthModel()
    bullet_1 = "Built an ETL pipeline using Airflow and Spark to process 10M+ rows daily."
    bullet_2 = "Worked on reporting stuff."
    bullet_3 = "Developed an SQL server using SQL and Python."

    print(f"Depth Score: {bullet_1} → {dm.predict_proba(bullet_1)}")
    print(f"Depth Score: {bullet_2} → {dm.predict_proba(bullet_2)}")
    print(f"Depth Score: {bullet_3} → {dm.predict_proba(bullet_3)}")