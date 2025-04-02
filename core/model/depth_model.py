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
        Loads the sentence-transformer model and the trained depth classifier.
        """
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.classifier = joblib.load(model_path)

    def predict_proba(self, bullet: str) -> float:
        """
        Returns probability that a bullet is 'deep'.
        """
        embedding = self.embedder.encode([bullet])
        return self.classifier.predict_proba(embedding)[0][1]

    def is_deep(self, bullet: str, threshold=0.5) -> bool:
        """
        Returns True if bullet is deeper than threshold.
        """
        return self.predict_proba(bullet) >= threshold

    def predict_batch(self, sentences: list[str]) -> list[int]:
        """
        Returns 0/1 predictions for a list of bullets.
        """
        embeddings = self.embedder.encode(sentences)
        return self.classifier.predict(embeddings)

    def analyze_batch(self, sentences: list[str]) -> dict:
        """
        Returns summary stats for a batch of resume bullets.
        """
        if not sentences:
            return {"deep_count": 0, "total_sentences": 0, "deep_percentage": 0}

        predictions = self.predict_batch(sentences)
        deep_count = sum(predictions)
        total = len(sentences)
        percent = (deep_count / total) * 100 if total else 0

        return {
            "deep_count": deep_count,
            "total_sentences": total,
            "deep_percentage": round(percent, 2),
        }

if __name__ == "__main__":
    dm = DepthModel()
    bullets = [
        "Built an ETL pipeline using Airflow and Spark to process 10M+ rows daily.",
        "Worked on reporting stuff.",
        "Developed an SQL server using SQL and Python."
    ]

    print("=== Individual Probabilities ===")
    for b in bullets:
        print(f"- {b} → {dm.predict_proba(b):.3f}")

    print("\n=== Batch Classification ===")
    for b, pred in zip(bullets, dm.predict_batch(bullets)):
        print(f"- {b} → {'🧠 Deep' if pred == 1 else '⚪ Shallow'}")

    print("\n=== Summary ===")
    print(dm.analyze_batch(bullets))
