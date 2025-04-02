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
from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from roles.job_roles import JOB_ROLES


# Load pre-trained models
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load the trained depth classifier
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/depth_model.pkl"))
depth_clf = joblib.load(MODEL_PATH)


def get_keywords_for_role(category: str, role: str):
    try:
        role_data = JOB_ROLES[category][role]
        keywords = role_data["required_skills"] + role_data["recommended_skills"]["technical"]
        return [kw.lower() for kw in keywords]
    except KeyError:
        return []


def get_deep_role_paragraph(category: str, role: str) -> str:
    """Returns a natural-language paragraph describing in-depth responsibilities for the role."""
    if category == "Software Development and Engineering" and role == "Backend Developer":
        return (
            "A backend developer is responsible for designing and maintaining server-side logic. "
            "They build RESTful APIs, manage database schemas, write business logic, and handle authentication. "
            "They often work with Python, Node.js, or Java, and deploy applications using Docker, CI/CD pipelines, and cloud platforms. "
            "Performance optimization, database tuning, and error handling are key responsibilities."
        )
    # Add more role descriptions as needed
    return "This role requires deep technical implementation, problem-solving, and system design skills."


def embed_role_context(category, role):
    # Extracted from real resume bullets (screenshot)
    deep_examples = [
        "Integrated a NoSQL distributed database into production software, improving performance by over 30%.",
        "Developed scripts to automate training data generation for machine learning models, boosting speed by 50%.",
        "Utilized parallelization and containerization on AWS EC2 to process large datasets efficiently.",
        "Executed analytics on 4 NoSQL databases including CouchDB and Aerospike, documenting configurations and performance.",
        "Built a backend with Node and Express to manage RESTful APIs and handle complex query logic.",
        "Trained an OpenAI-based AI assistant using LangChain and ChromaDB, tailored for browser context.",
        "Led chat feature implementation using Firebase and React Native, supporting real-time communication and scaling.",
        "Secured backend logic with Firebase authentication and Bearer token-based authorization flows.",
        "Integrated a Flask server backend using Python to simplify text using OpenAI's GPT-3.5.",
    ]

    embeddings = embedding_model.encode(deep_examples, convert_to_tensor=True)
    return embeddings.mean(dim=0)  # Averaged embedding represents the 'deep role context'


def score_bullet_ml(bullet: str) -> float:
    """Returns probability that bullet is 'deep' using trained model."""
    emb = embedding_model.encode([bullet])
    prob = depth_clf.predict_proba(emb)[0][1]  # probability of class 1 (deep)
    return round(prob, 3)


# Example usage
if __name__ == "__main__":
    category = "Software Development and Engineering"
    role = "Backend Developer"
    bullet = "Built a CI/CD pipeline using GitHub Actions and Docker, reducing deployment time by 60%."

    # Using the model to score the bullet
    depth_score = score_bullet_ml(bullet)
    print(f"Depth Score for bullet:\n{bullet}\n→ {depth_score}")