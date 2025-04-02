import re
from core.roles.job_roles import JOB_ROLES

def preprocess_text(text: str) -> str:
    """
    Normalize the bullet text by lowercasing all words and removing punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_keywords_for_role(role: str):
    """
    Retrieve the expected skills for a given role from JOB_ROLES.
    Searches through all categories and returns:
      - required_keywords: list of required skills
      - recommended_keywords: list of tech-related recommended skills
    If role is not found, returns empty lists.
    """
    for category, roles in JOB_ROLES.items():
        if role in roles:
            role_data = roles[role]
            required = role_data.get("required_skills", [])
            # Recommended skills may be nested under "recommended_skills" with a "technical" field.
            recommended = role_data.get("recommended_skills", {}).get("technical", [])
            return [kw.lower() for kw in required], [kw.lower() for kw in recommended]
    return [], []

def match_keywords(text: str, keywords: list) -> int:
    """
    Count the number of keywords that occur in the text.
    Using regex with word boundaries ensures matching whole words.
    """
    count = 0
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            count += 1
    return count

def evaluate_pattern(bullet: str, role: str) -> float:
    """
    Given a bullet point and a role (e.g., "Backend Developer"), compute a role relevance score.
    The score is based on the proportion of required and recommended keywords found.
    Returns a score between 0 and 1.
    """
    processed_bullet = preprocess_text(bullet)
    
    required_keywords, recommended_keywords = get_keywords_for_role(role)
    
    total_required = len(required_keywords)
    total_recommended = len(recommended_keywords)
    
    matched_required = match_keywords(processed_bullet, required_keywords) if total_required > 0 else 0
    matched_recommended = match_keywords(processed_bullet, recommended_keywords) if total_recommended > 0 else 0
    
    # Normalize each score. For example, if a role has 4 required keywords and we matched 2, then req_score = 0.5.
    req_score = matched_required / total_required if total_required > 0 else 0
    rec_score = matched_recommended / total_recommended if total_recommended > 0 else 0
    
    # Weight the scores (e.g., 70% required, 30% recommended)
    final_score = 0.7 * req_score + 0.3 * rec_score
    return round(final_score, 3)

if __name__ == "__main__":
    sample_bullet = "Developed RESTful APIs using Python and Flask, integrating with SQL databases."
    role = "Backend Developer"  # Adjust this to test with another role.
    score = evaluate_pattern(sample_bullet, role)
    print(f"Pattern Match Score for role '{role}': {score}")