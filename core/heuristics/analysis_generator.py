"""
Analysis Generator
------------------

This module generates a unified resume quality score by combining results from all three core heuristics:

    - **star_detection**    → Measures structural clarity (Situation-Task-Action-Result)
    - **depth_analysis**    → Measures technical/methodological richness of bullet points
    - **pattern_matcher**   → Measures alignment with job-role patterns and terminology

What This Module Does
---------------------
- Runs each heuristic on extracted resume bullet points
- Normalizes and aggregates heuristic outputs
- Applies dynamic weighting to each heuristic
- Outputs a **final composite resume quality score (0-1)**

What makes this different?
--------------------------
Scores are weighted using a logistic (sigmoid) function to ensure diminishing returns.
Essentially, as users reach a depth heuristic of 80%, the score approaches 1.0 but never quite reaches it.
This prevents overfitting to any one heuristic and ensures a balanced evaluation.


Structure
---------
- generate_resume_score(bullets)
    Main interface that runs all heuristics and returns a breakdown + final score.

- dynamic_weighted_score(depth, star, pattern)
    Applies sigmoid-based diminishing return logic to each heuristic before combining them.

- logistic_transform(x, x0, k)
    Smoothly transforms a heuristic score based on a defined target (`x0`) and steepness (`k`).

- __main__
    Contains test bullets for quick manual verification or experimentation.

TODO: Add a feedback generator segment.

"""


import sys
import os
import math

# Ensure Python can find 'core' and its submodules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.model.depth_model import DepthModel
from core.heuristics.star_detection import predict_star_sentences
from core.heuristics.pattern_matcher import predict_pattern_score

depth_model = DepthModel()

def generate_resume_score(bullets: list[str]) -> dict:
    """
    Runs all three heuristics on a list of resume bullets and returns
    a final composite resume score, as well as individual heuristic scores.
    """
    if not bullets:
        return {
            "star": 0,
            "depth": 0,
            "pattern": 0,
            "final_score": 0
        }

    # STAR Detection
    star_result = predict_star_sentences(bullets)
    star_score = star_result["star_percentage"] / 100  # 0–1

    # Depth Analysis
    depth_result = depth_model.analyze_batch(bullets)
    depth_score = depth_result["deep_percentage"] / 100  # 0–1

    # Pattern Matching
    pattern_score = predict_pattern_score(bullets)      # 0–1

    # Apply our logistic weighting
    final_score = dynamic_weighted_score(depth_score, star_score, pattern_score)

    return {
        "star": round(star_score, 3),
        "depth": round(depth_score, 3),
        "pattern": round(pattern_score, 3),
        "final_score": round(final_score, 3)
    }

def logistic_transform(x, x0=0.8, k=10):
    """
    Logistic (sigmoid) transform of x (0–1), saturating around x0.
    k controls steepness. Higher k => more abrupt near x0.
    """
    return 1.0 / (1.0 + math.exp(-k*(x - x0)))

def dynamic_weighted_score(depth, star, pattern):
    """
    Applies a logistic transform for each heuristic, then combines via weighted average.

    Example targets:
      - Depth saturates around 0.8 (80%)
      - STAR saturates around 0.6 (60%)
      - Pattern saturates around 0.9 (90%), or adjust as you see fit
    """
    # Transform each metric to produce diminishing returns
    depth_sig   = logistic_transform(depth, x0=0.8, k=10)   # saturates near 80%
    star_sig    = logistic_transform(star,  x0=0.6, k=12)   # saturates near 60%
    pattern_sig = logistic_transform(pattern, x0=0.9, k=10) # saturates near 90%

    # Weights for each metric (customize to your preference)
    w_depth   = 0.45
    w_star    = 0.35
    w_pattern = 0.20

    # Weighted sum of transformed scores
    return (
        depth_sig * w_depth +
        star_sig  * w_star +
        pattern_sig * w_pattern
    )


if __name__ == "__main__":
    sample_bullets = [
        "Developed CI/CD pipelines using GitHub Actions and Docker.",
        "Led a team of 3 engineers to deploy microservices to AWS.",
        "Handled some backend stuff.",
        "Built a dashboard for tracking KPIs."
    ]

    results = generate_resume_score(sample_bullets)

    print("=== Resume Score Breakdown ===")
    for k, v in results.items():
        print(f"{k.capitalize():<12} → {v}")
