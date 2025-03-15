"""
core/: Main application code.

data/: Sample resumes and training data.

model/: Model training and loading scripts.

tests/: Unit and integration tests.
"""

import core.extractor as extractor
from core.model.star_model import STARModel

def main():
    # Step 1: Extract text from resume
    file_path = "data/sample_good.pdf"
    text = extractor.extract_resume_text(file_path)

    # Step 2: Load STAR model
    model = STARModel()

    # Step 3: Detect STAR-compliant sentences
    sentences = text.split(". ")
    star_sentences = [sent for sent in sentences if model.predict(sent)]

    # Step 4: Print results
    print(f"Total Sentences: {len(sentences)}")
    print(f"STAR-Compliant Sentences: {len(star_sentences)}")
    print(f"STAR Percentage: {(len(star_sentences) / len(sentences)) * 100:.2f}%")

if __name__ == "__main__":
    main()