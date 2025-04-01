import pdfplumber
import re
import json
from collections import Counter
from typing import List, Dict, Optional

# ----------------------------------------------------------
#  Senior-Engineer-Level PDF Resume Extractor
#  This version tightens bullet detection, merges partial
#  lines more aggressively, and filters out “noise” lines.
#  The goal is fewer, more accurate bullets.
# ----------------------------------------------------------

# 1) Constants + Patterns
VALID_URL_ENDINGS = [
    ".com", ".org", ".net", ".io", ".co", ".co.uk", ".ai", ".edu",
    ".gov", ".us", ".uk", ".de", ".fr", ".jp", ".ca", ".au", ".info",
    ".dev", ".tech", ".biz", ".online", ".me", ".ly", ".app", ".cloud"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "protonmail.com", "aol.com", "mail.com"
]

COMMON_SECTION_HEADINGS = {
    # Work Experience
    "WORK EXPERIENCE", "EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY",
    "CAREER HISTORY", "WORK HISTORY", "PROFESSIONAL BACKGROUND",
    # Projects
    "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS", "TECHNICAL PROJECTS",
    "SELECTED PROJECTS", "KEY PROJECTS",
    # Education
    "EDUCATION", "ACADEMIC BACKGROUND", "EDUCATIONAL QUALIFICATIONS",
    "ACADEMIC QUALIFICATIONS", "DEGREES", "STUDIES",
    # Skills
    "SKILLS", "TECHNICAL SKILLS", "SOFT SKILLS", "LANGUAGES", "LANGUAGE SKILLS",
    "TECHNICAL COMPETENCIES", "CORE COMPETENCIES", "AREAS OF EXPERTISE",
    # Certifications and Training
    "CERTIFICATIONS", "TRAINING", "COURSES", "LICENSES", "PROFESSIONAL DEVELOPMENT",
    # Awards and Achievements
    "AWARDS", "HONORS", "ACHIEVEMENTS", "RECOGNITIONS", "ACCOMPLISHMENTS",
    # Hobbies and Interests
    "HOBBIES", "INTERESTS", "VOLUNTEER WORK", "EXTRACURRICULAR ACTIVITIES",
    # Publications and Research
    "PUBLICATIONS", "RESEARCH", "PRESENTATIONS", "CONFERENCES", "PAPERS",
    # Contact Information
    "REFERENCES", "CONTACT INFORMATION", "CONTACT", "CONTACT ME", "CONTACT DETAILS",
    "PERSONAL DETAILS", "HOW TO REACH ME",
    # Profile and Summary
    "PROFILE", "SUMMARY", "ABOUT ME", "OBJECTIVE", "GOALS", "MISSION",
    "PROFESSIONAL SUMMARY", "EXECUTIVE SUMMARY", "CAREER OBJECTIVE",
    # Links and Portfolios
    "LINKS", "PORTFOLIO", "SOCIAL MEDIA", "PROFILES", "ONLINE PRESENCE",
    # Additional Information
    "ADDITIONAL INFORMATION", "OTHER", "EXTRA", "MISCELLANEOUS",
    # Personal Details
    "PERSONAL DETAILS", "BIOGRAPHY", "RESUME", "CV",
    # Cover Letter
    "COVER LETTER", "LETTER",
    # Email and Phone
    "EMAIL", "PHONE", "TELEPHONE", "MOBILE",
    # Summaries
    "SKILLS SUMMARY", "SUMMARY OF QUALIFICATIONS", "SUMMARY OF EXPERIENCE",
    "SUMMARY OF SKILLS", "SUMMARY OF PROFILE", "SUMMARY OF BACKGROUND",
    "SUMMARY OF EDUCATION", "SUMMARY OF WORK", "SUMMARY OF EMPLOYMENT",
    "SUMMARY OF PROJECTS", "SUMMARY OF CERTIFICATIONS", "SUMMARY OF TRAINING",
    "SUMMARY OF AWARDS", "SUMMARY OF HOBBIES", "SUMMARY OF ACCOMPLISHMENTS",
    "SUMMARY OF ACHIEVEMENTS",
}
COMMON_SECTION_HEADINGS_LOWER = {h.lower() for h in COMMON_SECTION_HEADINGS}

DATE_PATTERN = re.compile(
    r"(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b)|"  # Month Year
    r"\b\d{4}\b|"                                                                  # Year only
    r"(?:\b(?:19|20)\d{2}\s*[–\-—]\s*(?:19|20)\d{2}\b)|"                           # Year range
    r"\b(?:19|20)\d{2}\s*-\s*Present\b|"                                           # Year-Present
    r"\b(?:Present|Current)\b|"                                                    # Present/Current
    r"\b(?:Summer|Winter|Fall|Spring)\s*(?:19|20)\d{2}\b"                          # Season Year
)

BULLET_PATTERNS = [
    r'^\s*[•▪♦➢➔⦿◦‣⁃■○✓]\s*',  # Various bullet chars
    r'^\s*[\*\-\+]\s+',         # Asterisk, hyphen, plus
    r'^\s*\d+[\.\)]\s*',        # Numbered list (1., 2) or 1)
    r'^\s*[a-z]\)\s*',          # Letter lists (a), b)
]
BULLET_DETECT_PATTERN = re.compile('|'.join(BULLET_PATTERNS))
REMOVE_BULLET_PATTERN = re.compile('|'.join(BULLET_PATTERNS))

# For deciding whether we should merge lines
INCOMPLETE_REGEX = re.compile(
    r'[^.!?]$|[,;:]$|(?:\b(?:and|or|but|with|for|to|the|a|an|in|on|at|by|using)\s*$)',
    re.IGNORECASE
)

# For final bullet merges in post-processing
INCOMPLETE_SENTENCE_PATTERN = re.compile(
    r'''
    (?:[^.!?]\s*$)|                  # Ends without punctuation
    (?:[a-z][^.!?]\s*$)|             # Ends with lowercase letter
    (?:[,;:]\s*$)|                   # Ends with comma/semicolon/colon
    (?:\b(?:and|or|but|with|for|to|the|a|an|in|on|at|by|as|using|via|through|while|when|where|who|which)\s*$)|
    (?:\b(?:including|such as|e\.g\.|i\.e\.|etc\.?)\s*$)|
    (?:\b(?:developed|created|built|implemented|designed|managed|led)\s*$)|
    (?:\b[a-z]+ing\s*$)|
    (?:\b[a-z]+ed\s*$)
    ''',
    re.IGNORECASE | re.VERBOSE
)

# ----------------------------------------------------------
# 2) Bullet / Line / Noise Detection
# ----------------------------------------------------------

def is_bullet_line(line: str) -> bool:
    """
    Returns True only if the line starts with a recognized
    bullet character, number, or letter pattern.
    """
    return bool(BULLET_DETECT_PATTERN.match(line))


def should_merge_lines(prev_line: str, current_line: str) -> bool:
    """
    Decide if the current_line should be appended to prev_line,
    i.e., if the previous bullet is 'incomplete' or this line
    looks like a continuation (starts lowercase, etc.).
    """
    if not prev_line:
        return False
    prev_line_stripped = prev_line.strip()

    # If the previous line ends with . ! or ?, it's probably done
    if re.search(r'[.!?]$', prev_line_stripped):
        return False

    # If the previous line is incomplete (no punctuation, ends with connector)
    if INCOMPLETE_REGEX.search(prev_line_stripped):
        return True

    # If the current line starts with lowercase, it might be continuing
    if current_line and current_line[0].islower():
        return True

    return False


def filter_noise(bullets: List[str]) -> List[str]:
    """
    Optional: Remove lines that look too short or have no
    meaningful words. Tweak to skip filler lines like
    "be the part of the little bud's life."
    """
    # Example "action" or "meaningful" words:
    action_verbs = {
        "created", "managed", "built", "developed", "implemented",
        "designed", "administered", "resolved", "provided", "led",
        "optimized", "deployed", "executed", "supported", "analyzed",
        "orchestrated", "planned"
    }
    filtered = []
    for b in bullets:
        words = b.lower().split()
        # If it's extremely short and has no strong verbs, skip
        if len(words) < 5 and not (action_verbs & set(words)):
            continue
        filtered.append(b)
    return filtered


# ----------------------------------------------------------
# 3) Text Extraction & Basic Parsing
# ----------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF while preserving line breaks.
    Handles hyphenated words (like "execute-\nion").
    """
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # Merge hyphenated words across line breaks
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                full_text.append(page_text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

    return "\n".join(full_text).strip()


def extract_contacts(text: str) -> Dict[str, List[str]]:
    """
    Extract emails and URLs. Then filter out junk (e.g. URL that is actually an email domain).
    """
    email_pattern = (
        r'(?:\b|mailto:)[a-zA-Z0-9._%+-]+@'
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    )
    url_pattern = r'''
        \b(?:https?://|www\.)
        [a-zA-Z0-9-]+
        (?:\.[a-zA-Z0-9-]+)+
        (?:/[^\s]*)?
        |
        \b[a-zA-Z0-9-]+\.
        (?:com|org|net|io|co|ai|edu|gov|us|uk|de|fr|jp|ca|au|info|dev|tech|biz)\b
    '''

    emails_found = re.findall(email_pattern, text, re.IGNORECASE)
    urls_found = re.findall(url_pattern, text, re.IGNORECASE | re.VERBOSE)

    filtered_urls = []
    for url in urls_found:
        url = url.strip().rstrip('/')
        if any(url.endswith(tld) or f".{tld}/" in url for tld in VALID_URL_ENDINGS):
            if not any(domain in url for domain in EMAIL_DOMAINS):
                if url not in filtered_urls:
                    filtered_urls.append(url)

    return {
        "emails": list({email.lower() for email in emails_found}),
        "links": filtered_urls
    }


# ----------------------------------------------------------
# 4) Section / Heading / Title Detection
# ----------------------------------------------------------

def is_section_heading(line: str) -> bool:
    """
    Check if a line is likely a resume section heading
    by comparing to known headings, case patterns, etc.
    """
    line_stripped = line.strip()
    line_lower = line_stripped.lower()

    # Check known headings
    if line_lower in COMMON_SECTION_HEADINGS_LOWER:
        return True

    # All uppercase heading
    if (
        line_stripped == line_stripped.upper()
        and len(line_stripped) > 3
        and any(c.isalpha() for c in line_stripped)
    ):
        return True

    # Pattern: short line ending with colon, e.g. "WORK EXPERIENCE:"
    if (
        len(line_stripped.split()) <= 4
        and (
            line_stripped.endswith(':')
            or re.search(r'^[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*:$', line_stripped)
        )
    ):
        return True

    return False


def is_job_title(line: str) -> bool:
    """
    Enhanced job title detection:
    - If the line contains date patterns
    - Or matches common job title patterns
    """
    line = line.strip()
    # If date found, we often treat it as a job or position block
    if DATE_PATTERN.search(line):
        return True

    patterns = [
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:at|@)\s+',   # "Position at Company"
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+',             # "Position, Company"
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[–\-—]\s+',     # "Position - Company"
        r'^[A-Z][a-zA-Z0-9\s&]+\s*[•\-|]\s*',               # "Company • Position"
        r'^(?:Senior|Junior|Lead|Principal)\s+[A-Z][a-z]+', # "Senior Developer"
        r'^[A-Z][a-z]+\s+(?:Engineer|Developer|Manager|Specialist|Analyst|Designer)\b',
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Team|Group|Department)\b'
    ]
    return any(re.search(pattern, line) for pattern in patterns)


def extract_sections(text: str) -> List[Dict]:
    """
    Split the resume text by lines, detect section headings,
    job titles, and group lines into structured sections.
    """
    lines = text.split("\n")
    sections = []
    current_section = None
    current_entry = None
    previous_line_was_heading = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            previous_line_was_heading = False
            continue

        # If it's a recognized heading, start a new section
        if is_section_heading(line):
            if current_section:  # close out old section
                if current_entry:
                    current_section["entries"].append(current_entry)
                    current_entry = None
                sections.append(current_section)

            current_section = {
                "section_name": line,
                "entries": []
            }
            previous_line_was_heading = True
            continue

        # If we haven't yet started a section, skip lines until we do
        if not current_section:
            continue

        # If it's a job title or we just had a heading, start a new "entry"
        if is_job_title(line) or previous_line_was_heading:
            if current_entry:
                current_section["entries"].append(current_entry)

            current_entry = {
                "title": line,
                "bullets": []
            }
            previous_line_was_heading = False
            continue

        # Otherwise, add line to current entry's bullets
        if current_entry:
            current_entry["bullets"].append(line)
        else:
            # Orphaned line (no job title yet) => create an "Additional" entry
            current_entry = {
                "title": "Additional Information",
                "bullets": [line]
            }

        previous_line_was_heading = False

    # Final close-out
    if current_section:
        if current_entry:
            current_section["entries"].append(current_entry)
        sections.append(current_section)

    return sections


# ----------------------------------------------------------
# 5) Merging Bullets, Removing Noise
# ----------------------------------------------------------

def merge_multiline_bullets(bullet_lines: List[str]) -> List[str]:
    """
    Merge lines that do not start with bullet characters and
    appear to be continuations (using `should_merge_lines`).
    Then optionally filter out “noise” lines.
    """
    merged_bullets = []
    current_bullet = ""

    for line in bullet_lines:
        line = line.rstrip()
        if not line:
            continue

        # Check if line is a new bullet
        is_bul = is_bullet_line(line)
        clean_line = REMOVE_BULLET_PATTERN.sub("", line).strip() if is_bul else line

        if is_bul:
            # If we have an in-progress bullet, store it
            if current_bullet:
                merged_bullets.append(current_bullet.strip())
            current_bullet = clean_line
        else:
            # Possibly merge into the previous bullet
            if current_bullet and should_merge_lines(current_bullet, clean_line):
                current_bullet += " " + clean_line
            else:
                # Start a new bullet if old bullet is done
                if current_bullet:
                    merged_bullets.append(current_bullet.strip())
                current_bullet = clean_line

    # Wrap up last bullet
    if current_bullet:
        merged_bullets.append(current_bullet.strip())

    # Filter out extremely short/noise lines if desired
    merged_bullets = filter_noise(merged_bullets)
    return merged_bullets


# ----------------------------------------------------------
# 6) Skill Extraction
# ----------------------------------------------------------

def extract_skills_from_pdf(pdf_path: str, custom_skill_list: Optional[List[str]] = None) -> List[str]:
    """
    Return top matched + potential skills from text. 
    We look for known tech skills and also guess from 
    frequent noun phrases (up to 3 words).
    """
    TECH_SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
        'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'perl',
        'haskell', 'elixir', 'clojure', 'dart', 'html', 'css', 'sass', 'less',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'sqlite', 'django',
        'flask', 'react', 'angular', 'vue', 'node', 'express', 'spring',
        'laravel', 'rails', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'terraform', 'ansible', 'puppet', 'chef', 'git', 'jenkins', 'ci/cd',
        'linux', 'bash', 'powershell', 'windows server', 'macos', 'machine learning',
        'ai', 'data science', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'keras',
        'agile', 'scrum', 'devops', 'oop', 'rest', 'api', 'microservices',
        'graphql', 'grpc', 'tableau', 'power bi', 'excel', 'word', 'powerpoint',
        'outlook', 'jira', 'confluence'
    }

    STOP_WORDS = {
        'and', 'the', 'for', 'with', 'you', 'are', 'but', 'have', 'has', 'had',
        'this', 'that', 'these', 'those', 'from', 'their', 'will', 'would',
        'been', 'they', 'which', 'your', 'when', 'where', 'what', 'who', 'how',
        'should', 'could', 'would', 'might', 'must', 'shall', 'can', 'may'
    }

    # Merge custom skills if any
    if custom_skill_list:
        TECH_SKILLS.update({s.lower() for s in custom_skill_list})

    # Extract text
    try:
        text = extract_text_from_pdf(pdf_path).lower()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    found_skills = set()
    # 1) Direct match for known skills
    for skill in TECH_SKILLS:
        if ' ' in skill:
            if skill in text:
                found_skills.add(skill)
        else:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.add(skill)

    # 2) Heuristic search for short noun phrases 
    # (up to 3 words) that might be skills
    noun_phrases = re.findall(r'\b(?:[a-z]+\s){0,3}[a-z]+\b', text)
    potential_skills = []
    for phrase in noun_phrases:
        # Exclude if in stop words, or already recognized
        if phrase in found_skills:
            continue
        tokens = phrase.split()
        if len(tokens) <= 3 and not any(t in STOP_WORDS for t in tokens):
            potential_skills.append(phrase)

    # 3) Frequency scoring for potential skills
    freq_counter = Counter(potential_skills)
    top_candidates = [skill for skill, _ in freq_counter.most_common(20)]

    # Sort known skills by frequency in text, then add top 10 guessed
    sorted_known = sorted(found_skills, key=lambda s: (-text.count(s), s))
    sorted_known.extend(top_candidates[:10])

    return sorted_known[:20]


# ----------------------------------------------------------
# 7) Post-Processing (Merging Incomplete Sentences)
# ----------------------------------------------------------

def post_process_resume_data(resume_data: Dict) -> Dict:
    """
    Final pass: standardize emails, refine bullet merges 
    for incomplete sentences.
    """
    # Clean contacts
    resume_data['contacts']['emails'] = [email.lower() for email in resume_data['contacts']['emails']]

    # For each bullet, if it ends with typical incomplete pattern, merge with next
    for section in resume_data['sections']:
        section['section_name'] = section['section_name'].strip()

        for entry in section['entries']:
            entry['title'] = entry['title'].strip()

            cleaned_bullets = []
            current_sentence = ""

            for bullet in entry['bullets']:
                bullet = bullet.strip()
                if not bullet:
                    continue

                # If the current sentence is incomplete, then merge
                if current_sentence and INCOMPLETE_SENTENCE_PATTERN.search(current_sentence):
                    current_sentence += " " + bullet
                else:
                    # If we had a complete bullet, store it
                    if current_sentence:
                        cleaned_bullets.append(current_sentence)
                    current_sentence = bullet

            if current_sentence:
                cleaned_bullets.append(current_sentence)

            entry['bullets'] = cleaned_bullets

    return resume_data


# ----------------------------------------------------------
# 8) Main
# ----------------------------------------------------------

if __name__ == "__main__":
    pdf_path = "core/data/sample_good.pdf"

    try:
        pdf_text = extract_text_from_pdf(pdf_path)

        # Gather contacts (emails/links)
        contacts = extract_contacts(pdf_text)

        # Extract hierarchical sections
        resume_sections = extract_sections(pdf_text)

        # Merge bullets in each entry
        for section in resume_sections:
            for entry in section["entries"]:
                entry["bullets"] = merge_multiline_bullets(entry["bullets"])

        # Build final output
        output = {
            "contacts": contacts,
            "sections": resume_sections,
            "skills": extract_skills_from_pdf(pdf_path)
        }

        # Post-process to unify incomplete sentences, etc.
        output = post_process_resume_data(output)

        # Print JSON
        print(json.dumps(output, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error processing resume: {e}")
