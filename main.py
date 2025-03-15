import core.extractor as extractor

def main():
    file_path = "data/sample_good.pdf"
    text = extractor.extract_resume_text(file_path)
    # print("=== Extracted Text ===")
    # print(text)

    # Now extract emails and links
    contacts = extractor.extract_contacts(text)
    print("\n=== Detected Emails & Links ===")
    print("Emails:", contacts["emails"])
    print("Links:", contacts["links"])

if __name__ == "__main__":
    main()
