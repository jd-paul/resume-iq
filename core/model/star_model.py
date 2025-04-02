"""
star_model.py

Wrapper for the trained model (star_model.pk) and vectorizer (star_vectorizer.pk).

This allows the rest of the Resume IQ system to easily evaluate bullet points for technical and methodological depth.
"""

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
        self.model = joblib.load("core/model/star_model.pkl")
        self.vectorizer = joblib.load("core/model/star_vectorizer.pkl")


    def predict(self, text):
        """
        Predicts whether a sentence is STAR-compliant.
        """
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)
        return prediction[0] == 1  # Returns True if STAR-compliant, False otherwise

if __name__ == "__main__":
    model = STARModel()
    print(model.predict("Developed a system that increased efficiency by 30%."))
    print(model.predict("Worked on various projects without clear outcomes."))