"""

The STAR method is a structured manner of responding to a behavioral-based interview question by discussing the
specific situation, task, action, and result of the situation you are describing.

There are many ways to implement analysis of this on bullet points in a resume. The approached used here
is Logistic Regression classifier that determines whether a sentence follows the STAR method or not.

"""

import joblib
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from core.extractor import extract_text_from_pdf, extract_sections, merge_multiline_bullets

# Load the trained STAR model and vectorizer
MODEL_PATH = "core/model/star_model.pkl"
VECTORIZER_PATH = "core/model/star_vectorizer.pkl"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_star_sentences(sentences):
    """
    Evaluates whether each sentence follows the STAR method.

    Args:
        sentences (list of str): List of sentences from the resume.

    Returns:
        dict: STAR classification results with STAR percentage.
    """
    if not sentences:
        return {"star_count": 0, "total_sentences": 0, "star_percentage": 0}

    # Convert sentences to TF-IDF features
    sentences_vectorized = vectorizer.transform(sentences)

    # Predict STAR classification (1 = STAR, 0 = Not STAR)
    predictions = model.predict(sentences_vectorized)

    # Count STAR sentences
    star_count = sum(predictions)
    total_sentences = len(sentences)
    star_percentage = (star_count / total_sentences) * 100 if total_sentences else 0

    return {
        "star_count": star_count,
        "total_sentences": total_sentences,
        "star_percentage": round(star_percentage, 2)
    }

if __name__ == "__main__":
    pdf_path = "core/data/sample_bad.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)

    resume_sections = extract_sections(pdf_text) # Extract structured sections

    # Merge multi-line bullets in extracted sections
    all_bullets = []
    for section in resume_sections:
        for entry in section["entries"]:
            cleaned_bullets = merge_multiline_bullets(entry["bullets"])
            all_bullets.extend(cleaned_bullets)

    # Run STAR detection
    results = predict_star_sentences(all_bullets)

    # Print results
    print("\n=== STAR Method Detection Results ===")
    print(f"Total Sentences: {results['total_sentences']}")
    print(f"STAR Sentences: {results['star_count']}")
    print(f"STAR Percentage: {results['star_percentage']}%\n")

    # Print classification for each bullet point
    print("=== Bullet Point Classification ===")
    for bullet in all_bullets:
        prediction = model.predict(vectorizer.transform([bullet]))[0]
        classification = "✅ STAR" if prediction == 1 else "❌ Not STAR"
        print(f"- {bullet} → {classification}")
