"""
STAR Method Heuristic
---------------------

This module evaluates each bullet point (or sentence) of a resume according to the STAR methodology:
Situation, Task, Action, and Result. It determines whether a statement includes a structured narrative
of what happened, why it happened, what the candidate did, and the measurable outcome.

1. Situation
   - Sets the context or background of the scenario.
   - For instance, "In my previous role at X, we faced significant deployment delays..."

2. Task
   - Specifies the challenge or objective, typically the candidate's goal or responsibility.
   - Example: "My task was to streamline and automate the deployment process..."

3. Action
   - Details the candidate's personal contributions or steps taken to address the task.
   - Example: "I introduced a CI/CD pipeline using Jenkins and scripted Docker builds..."

4. Result
   - Describes the outcomes or impacts of the actions taken.
   - Example: "...which reduced deployment time by 60% and increased reliability."

Implementation
--------------
- Logistic Regression Classifier:
  - Trained on labeled sentences or bullets (tagged STAR vs. non-STAR).
  - Predicts the probability that a bullet follows the STAR framework.
  - Can be refined by incorporating sublabels for each component (S/T/A/R) or by using multi-class classification.

Comparison to Depth Analysis
----------------------------
- The STAR heuristic focuses on the *structure* of a bullet—whether it describes a Situation, Task, Action, and Result.
- Depth Analysis evaluates *how much technical/methodological detail* is provided, rather than the presence of S, T, A, R.
- A bullet can score well on STAR without being extremely technical, and vice versa.

Usage
-----
After extracting bullet points from the resume, each bullet is classified as STAR-compliant or not.
This result is factored into the overall scoring of the resume, often with a weighted approach along
with other heuristics (Depth, Role Relevance, etc.).
"""

import joblib
import sys
import os

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