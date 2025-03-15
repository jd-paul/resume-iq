import pdfplumber
import docx
import re

VALID_URL_ENDINGS = [
    ".com", ".org", ".net", ".io", ".co", ".co.uk", ".ai", ".edu",
    ".gov", ".us", ".uk", ".de", ".fr", ".jp", ".ca", ".au", ".info",
    ".dev", ".tech", ".biz", ".online"
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

def extract_contacts(text):
    """
    Extracts emails and valid URLs from the resume text.
    """
    # ✅ Email pattern
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'

    # ✅ URL pattern
    url_pattern = r'\b(?:https?://|www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?\b'

    emails_found = re.findall(email_pattern, text)
    urls_found = re.findall(url_pattern, text)

    # 🔍 Debug: Print raw extracted URLs before filtering
    print("\n=== Raw Extracted URLs ===")
    print(urls_found)

    # ✅ Filter URLs:
    filtered_urls = []
    for url in urls_found:
        if any(url.endswith(tld) or f".{tld}/" in url for tld in VALID_URL_ENDINGS):
            # ❌ Ignore email domains (e.g., gmail.com, outlook.com)
            if not any(url == domain for domain in EMAIL_DOMAINS):
                filtered_urls.append(url)

    return {
        "emails": emails_found,
        "links": list(set(filtered_urls))  # Remove duplicates
    }



def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_resume_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Currently only .pdf and .docx are handled.")
