import pandas as pd
import re

CSV_PATH = "core/data/resume.csv"
OUTPUT_FILE = "core/data/unlabeled_bullets.txt"

def cleanup_line(line: str) -> str:
    """
    Removes trailing date patterns, references to 'Company Name', 'Education', 'Firms', etc.
    so that we keep only the actual bullet portion.
    """

    # 1) Remove trailing date patterns like "06/2013   to   02/2016"
    #    or single date references like "07/2015"
    line = re.sub(r'(0[1-9]|1[0-2])/\d{4}\s*to\s*(0[1-9]|1[0-2])/\d{4}.*', '', line)
    line = re.sub(r'(0[1-9]|1[0-2])/\d{4}.*', '', line)

    # 2) Remove trailing references to "Company Name" and anything after
    line = re.split(r'Company Name\s*.*', line)[0]

    # 3) Remove trailing "Education" section references
    line = re.split(r'Education\s*.*', line)[0]

    # 4) Remove trailing "Firms" references if any
    line = re.split(r'Firms?\s*[:]?.*', line)[0]

    # 5) Remove extra spacing
    line = line.strip()

    return line


def extract_chunks(text: str):
    """
    Splits the resume text on newlines or bullet-like symbols,
    then cleans up each chunk to remove extraneous references.
    """
    # Split on newlines or bullet-like separators
    chunks = re.split(r'[\n•●\-\–]+', text)
    clean_chunks = []

    for chunk in chunks:
        # Trim whitespace
        chunk = chunk.strip()

        # Apply the second-pass cleanup
        chunk = cleanup_line(chunk)

        # Heuristic: Keep lines that are at least 6 words
        # AND start with an action verb or capital letter
        if len(chunk.split()) >= 6 and re.match(
            r"^(Managed|Developed|Led|Created|Handled|Organized|Coordinated|Designed|Built|Worked|Conducted|Implemented|Executed|Assisted|Served|Enhanced|Drafted|Reduced|Increased|Collaborated|Partnered|Presented|Improved|Wrote|Participated|Defined|Automated|Integrated|Supervised|Monitored|Resolved|Recommended|Promoted|Maintained|Analyzed|Directed|Prepared|Reviewed)",
            chunk
        ):
            clean_chunks.append(chunk)

    return clean_chunks


def extract_from_resume_csv():
    df = pd.read_csv(CSV_PATH)

    all_bullets = []

    # We assume the second column has the raw resume text
    for _, row in df.iterrows():
        resume_text = str(row[1])
        bullets = extract_chunks(resume_text)
        all_bullets.extend(bullets)

    # Remove duplicates & sort for consistency
    unique_bullets = sorted(set(all_bullets))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for bullet in unique_bullets:
            f.write(bullet + "\n")

    print(f"✅ Extracted {len(unique_bullets)} cleaner bullets to {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_from_resume_csv()
