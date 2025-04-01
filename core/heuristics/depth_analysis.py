"""

The depth analysis module is a one of the core heuristics in analysing resumes in this program. It analyses each bullet point,
and evaluates whether the bullet point goes in-depth and focuses on the methodology of their role, or if it
is too generic and lacks depth.

The method used for this is...

"""

import joblib
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from roles import JOB_ROLES

def get_keywords_for_role(category: str, role: str):
    try:
        role_data = JOB_ROLES[category][role]
        keywords = role_data["required_skills"] + role_data["recommended_skills"]["technical"]
        return [kw.lower() for kw in keywords]
    except KeyError:
        return []
