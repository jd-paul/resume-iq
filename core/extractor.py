import pdfplumber
import re
import json

VALID_URL_ENDINGS = [
    ".com", ".org", ".net", ".io", ".co", ".co.uk", ".ai", ".edu",
    ".gov", ".us", ".uk", ".de", ".fr", ".jp", ".ca", ".au", ".info",
    ".dev", ".tech", ".biz", ".online"
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

COMMON_SECTION_HEADINGS = {
    # Work Experience
    "WORK EXPERIENCE", "EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY",
    # Projects
    "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS",
    # Education
    "EDUCATION", "ACADEMIC BACKGROUND", "EDUCATIONAL QUALIFICATIONS",
    # Skills
    "SKILLS", "TECHNICAL SKILLS", "SOFT SKILLS", "LANGUAGES",
    # Certifications and Training
    "CERTIFICATIONS", "TRAINING", "COURSES",
    # Awards and Achievements
    "AWARDS", "HONORS", "ACHIEVEMENTS",
    # Hobbies and Interests
    "HOBBIES", "INTERESTS", "VOLUNTEER WORK",
    # Publications and Research
    "PUBLICATIONS", "RESEARCH", "PRESENTATIONS",
    # Contact Information
    "REFERENCES", "CONTACT INFORMATION", "CONTACT", "CONTACT ME", "CONTACT DETAILS",
    # Profile and Summary
    "PROFILE", "SUMMARY", "ABOUT ME", "OBJECTIVE", "GOALS", "MISSION",
    "PROFESSIONAL SUMMARY", "EXECUTIVE SUMMARY",
    # Links and Portfolios
    "LINKS", "PORTFOLIO", "SOCIAL MEDIA", "PROFILES",
    # Additional Information
    "ADDITIONAL INFORMATION", "OTHER", "EXTRA",
    # Personal Details
    "PERSONAL DETAILS", "BIOGRAPHY", "RESUME",
    # Cover Letter
    "COVER LETTER", "LETTER",
    # Email and Phone
    "EMAIL", "PHONE",
    # Summaries
    "SKILLS SUMMARY", "SUMMARY OF QUALIFICATIONS", "SUMMARY OF EXPERIENCE",
    "SUMMARY OF SKILLS", "SUMMARY OF PROFILE", "SUMMARY OF BACKGROUND",
    "SUMMARY OF EDUCATION", "SUMMARY OF WORK", "SUMMARY OF EMPLOYMENT",
    "SUMMARY OF PROJECTS", "SUMMARY OF CERTIFICATIONS", "SUMMARY OF TRAINING",
    "SUMMARY OF AWARDS", "SUMMARY OF HOBBIES", "SUMMARY OF ACCOMPLISHMENTS",
    "SUMMARY OF ACHIEVEMENTS",
}
COMMON_SECTION_HEADINGS_LOWER = set(h.lower() for h in COMMON_SECTION_HEADINGS)

# Regex to detect date ranges
DATE_PATTERN = re.compile(
    r"(19|20)\d{2}"                  # 4-digit year starting 19xx or 20xx
    r"(\s?[–\-—]\s?(19|20)\d{2})?"   # optional dash + second year
    r"|(19|20)\d{2}\s?-\s?Present"   # or “2023-Present”
    r"|Summer\s?(19|20)\d{2}"        # or “Summer 2024”
)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF while preserving line breaks.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def extract_contacts(text):
    """
    Extracts emails and valid URLs from the resume text.
    """
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    url_pattern = r'\b(?:https?://|www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b'

    emails_found = re.findall(email_pattern, text)
    urls_found = re.findall(url_pattern, text)

    filtered_urls = []
    for url in urls_found:
        if any(url.endswith(tld) or f".{tld}/" in url for tld in VALID_URL_ENDINGS) or "/" in url:
            if not any(url == domain for domain in EMAIL_DOMAINS):
                filtered_urls.append(url)

    return {
        "emails": list(set(emails_found)),
        "links": list(set(filtered_urls))
    }

def is_section_heading(line):
    """
    Treat the line as a section heading if:
    1) Its lower-cased version appears in COMMON_SECTION_HEADINGS_LOWER, or
    2) It's fully uppercase (length > 3).
    """
    line_stripped = line.strip()
    line_lower = line_stripped.lower()

    if line_lower in COMMON_SECTION_HEADINGS_LOWER:
        return True

    # Also treat lines that are fully uppercase (and not just one or two chars) as headings
    if line_stripped == line_stripped.upper() and len(line_stripped) > 3:
        return True

    return False

def is_job_title(line):
    """
    Heuristic check for lines that likely describe a position/company + date range.
    """
    return bool(DATE_PATTERN.search(line))

def extract_sections(text):
    """
    Extracts structured sections from the resume text.
    Returns something like:
    [
      {
        "section_name": "WORK EXPERIENCE",
        "entries": [
          {
            "title": "King's Labs - Software Engineer • 2024 - 2025",
            "bullets": ["..."]
          },
          ...
        ]
      },
      ...
    ]
    """
    lines = text.split("\n")

    sections = []
    current_section = None
    current_entry = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if is_section_heading(line):
            # Close out previous section
            if current_section:
                if current_entry:
                    current_section["entries"].append(current_entry)
                    current_entry = None
                sections.append(current_section)

            # Start a new section
            current_section = {"section_name": line, "entries": []}
            continue

        # If there's no current section, skip until we find one
        if not current_section:
            continue

        if is_job_title(line):
            # Close out the previous entry
            if current_entry:
                current_section["entries"].append(current_entry)
            # Start a new entry
            current_entry = {"title": line, "bullets": []}
            continue

        if current_entry:
            current_entry["bullets"].append(line)

    # Final wrap-up
    if current_section:
        if current_entry:
            current_section["entries"].append(current_entry)
        sections.append(current_section)

    return sections

BULLET_DETECT_PATTERN = re.compile(r'^\s*[•*\-]\s*')
REMOVE_BULLET_PATTERN = re.compile(r'^\s*[•\-\*]\s*')

def merge_multiline_bullets(bullet_lines):
    """
    Merges consecutive lines into a single bullet if they don't start with a bullet symbol.
    Then strips the leading bullet symbol from each bullet.
    """
    merged_bullets = []
    current_bullet = ""

    for line in bullet_lines:
        # If the line *appears* to start with a bullet symbol, we treat it as a new bullet.
        if BULLET_DETECT_PATTERN.match(line):
            # If we already have a bullet being built, append it
            if current_bullet:
                merged_bullets.append(current_bullet.strip())
            current_bullet = line
        else:
            # Otherwise, we continue the existing bullet
            current_bullet += " " + line

    # Append the final bullet if it exists
    if current_bullet:
        merged_bullets.append(current_bullet.strip())

    # Now remove the bullet symbol
    cleaned_bullets = [
        REMOVE_BULLET_PATTERN.sub("", bullet).strip()
        for bullet in merged_bullets
    ]
    return cleaned_bullets

if __name__ == "__main__":
    pdf_path = "core/data/sample_good.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    contacts = extract_contacts(pdf_text)
    resume_sections = extract_sections(pdf_text)

    # Merge multi-line bullets, removing bullet symbols
    for section in resume_sections:
        for entry in section["entries"]:
            entry["bullets"] = merge_multiline_bullets(entry["bullets"])

    output = {
        "contacts": contacts,
        "sections": resume_sections
    }

    # Print with ensure_ascii=False so that special characters remain readable
    print(json.dumps(output, indent=2, ensure_ascii=False))
