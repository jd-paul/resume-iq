import core.extractor as extractor

def main():
    file_path = "data/sample_good.pdf"
    text = extractor.extract_resume_text(file_path)

if __name__ == "__main__":
    main()
