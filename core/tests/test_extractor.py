import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Now import the correct functions
from core.extractor import extract_resume_text, extract_bullet_points

# Path to your sample resume
file_path = "core/data/sample_good.pdf"

# Extract text and bullet points
resume_text = extract_resume_text(file_path)
bullet_points = extract_bullet_points(resume_text)

# Print results
print("=== Extracted Bullet Points ===")
for bullet in bullet_points:
    print(f"- {bullet}")
