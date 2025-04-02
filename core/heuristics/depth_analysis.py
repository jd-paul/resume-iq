"""
Depth Analysis Heuristic
------------------------

This module evaluates each bullet point of a resume for its "depth," i.e., whether it conveys
substantial technical or methodological detail. Unlike the STAR heuristic, which checks for
Situation, Task, Action, and Result structure, Depth Analysis specifically focuses on:

1. Clear, Specific Actions
   - Uses strong action verbs ("Implemented," "Automated," "Deployed," etc.)
   - Mentions domain-relevant technologies or methods ("Docker + K8s," "TensorFlow," "CI/CD pipelines")

2. Methodology
   - Outlines *how* and *why* the action was performed
   - Provides context for the approach or rationale
   - For instance, "Automated CI/CD pipelines using Jenkins to streamline releases"
     is deeper than "Worked on CI/CD."

3. Technical Detail and Contextual Richness
   - Indicates the *substance* or *impact* of the achievement ("Reduced deployment time by 60%")
   - Goes beyond generic statements ("Handled various tasks") to demonstrate ownership, complexity,
     or significant problem-solving

Importantly:
- This heuristic does NOT measure role-relevance. It doesn't matter whether the candidate is
  applying for backend engineering, data science, or something else.
- Instead, it universally checks if bullet points are *methodologically detailed* and *technically* or
  *contextually* robust.

Implementation
--------------
1. Embedding + Classifier
   - We train or use a classifier on top of pre-trained sentence embeddings (e.g., MPNet, DistilBERT),
     labeling bullet points as "deep" vs "shallow."
   - This approach demonstrates real ML skill and can generalize across multiple domains once we
     have enough labeled examples.

2. Simple Heuristic via Cosine Similarity
   - Compare user bullets to a curated list of "deep" example bullets (embedding-based).
   - Less accurate than a trained model, but quicker to implement if you have few labeled samples.

Usage
-----
Each bullet receives a 'depth score' (0-1). We then factor that score in the analysis_generator.py
"""

import joblib
import sys
import os
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from core.extractor import extract_text_from_pdf, extract_sections, merge_multiline_bullets

# Load sentence transformer and trained classifier
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
MODEL_PATH = "core/model/depth_model.pkl"
depth_clf = joblib.load(MODEL_PATH)

def predict_depth_sentences(sentences):
    """
    Evaluates whether each sentence is methodologically deep.

    Args:
        sentences (list of str): List of sentences from the resume.

    Returns:
        dict: Depth classification results with depth percentage.
    """
    if not sentences:
        return {"deep_count": 0, "total_sentences": 0, "deep_percentage": 0}

    embeddings = EMBEDDING_MODEL.encode(sentences)
    predictions = depth_clf.predict(embeddings)

    deep_count = sum(predictions)
    total_sentences = len(sentences)
    deep_percentage = (deep_count / total_sentences) * 100 if total_sentences else 0

    return {
        "deep_count": deep_count,
        "total_sentences": total_sentences,
        "deep_percentage": round(deep_percentage, 2)
    }

if __name__ == "__main__":
    pdf_path = "core/data/sample_good.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)
    resume_sections = extract_sections(pdf_text)

    all_bullets = []
    for section in resume_sections:
        for entry in section["entries"]:
            cleaned_bullets = merge_multiline_bullets(entry["bullets"])
            all_bullets.extend(cleaned_bullets)

    results = predict_depth_sentences(all_bullets)

    print("\n=== Depth Analysis Results ===")
    print(f"Total Sentences: {results['total_sentences']}")
    print(f"Deep Sentences: {results['deep_count']}")
    print(f"Depth Percentage: {results['deep_percentage']}%\n")

    print("=== Bullet Point Classification ===")
    for bullet in all_bullets:
        embedding = EMBEDDING_MODEL.encode([bullet])
        prediction = depth_clf.predict(embedding)[0]
        classification = "🧠 Deep" if prediction == 1 else "⚪ Shallow"
        print(f"- {bullet} → {classification}")
