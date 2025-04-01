import os
import sys
import glob

# Ensure Python sees extractor.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extractor import (
    extract_text_from_pdf,
    extract_sections,
)

PDF_FOLDER = "core/data/resume_pdfs/INFORMATION-TECHNOLOGY"
OUTPUT_FILE = "core/data/unlabeled_bullets.txt"

def parse_folder_of_pdfs(pdf_folder: str, output_file: str):
    all_bullets = []

    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_folder}")
        return

    for pdf_path in pdf_paths:
        print(f"Parsing: {pdf_path}")
        try:
            # 1) Extract raw text
            pdf_text = extract_text_from_pdf(pdf_path)
            # 2) Convert raw text to structured sections
            resume_sections = extract_sections(pdf_text)

            # 3) Gather bullets
            for section in resume_sections:
                for entry in section.get("entries", []):
                    for bullet in entry.get("bullets", []):
                        bullet = bullet.strip()
                        # Filter out short lines (< 6 words, for example)
                        if len(bullet.split()) >= 6:
                            all_bullets.append(bullet)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    # 4) Deduplicate & sort
    unique_bullets = sorted(set(all_bullets))

    # 5) Write to unlabeled_bullets.txt
    with open(output_file, "w", encoding="utf-8") as f:
        for bullet in unique_bullets:
            f.write(bullet + "\n")

    print(f"✅ Extracted {len(unique_bullets)} bullets from folder: {pdf_folder} → {output_file}")

if __name__ == "__main__":
    parse_folder_of_pdfs(PDF_FOLDER, OUTPUT_FILE)
