"""
core/: Main application code.

data/: Sample resumes and training data.

model/: Model training and loading scripts.

tests/: Unit and integration tests.
"""

"""

Program flowchart
    A: User Uploads Resume PDF
    B: Extract Text from PDF
    C: Heuristics Evaluation
    D: User receives score and feedback
"""

import core.extractor as extractor
from core.model.star_model import STARModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI()

# Enable CORS so your Next.js frontend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your specific frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-skills")
async def extract_skills(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        skills = extractor.extract_skills_from_pdf(temp_path)
        return {"skills": skills}

    except Exception as e:
        return {"error": str(e)}


def main():
    # Step 1: Extract text from resume
    pdf_path = "core/data/sample_good.pdf"
    text = extractor.extract_text_from_pdf(pdf_path)

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
