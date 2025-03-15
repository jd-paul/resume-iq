import joblib

class STARModel:
    """
    Loads a pre-trained STAR detection model (star_model.pkl) and vectorizer (star_vectorizer.pkl).
    This used to analyze whether a sentence follows the STAR method.

        star_model.pkl - Contains the trained Logistic Regression model.
        star_vectorizer.pkl - Contains the TF-IDF vectorizer used to convert text into numerical features.
    """

    def __init__(self, model_path="model/star_model.pkl", vectorizer_path="model/star_vectorizer.pkl"):
        """
        Initializes the STAR model and vectorizer.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """
        Predicts whether a sentence is STAR-compliant.
        """
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)
        return prediction[0] == 1  # Returns True if STAR-compliant, False otherwise

if __name__ == "__main__":
    # Example usage
    model = STARModel()
    print(model.predict("Developed a system that increased efficiency by 30%."))  # Should return True
    print(model.predict("Worked on various projects without clear outcomes."))  # Should return False