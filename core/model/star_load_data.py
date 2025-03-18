import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/star_data.txt")

def load_data(file_path):
    """
    Loads labeled data from a text file.
    Each line should be in the format: "sentence | label"
    """
    data = pd.read_csv(file_path, sep="|", header=None, names=["text", "label"])
    return data

def preprocess_data(data):
    """
    Preprocesses the data:
    - Converts labels to integers.
    - Drops rows with missing values.
    """
    data["label"] = data["label"].astype(int)  # Conveart labels to integers
    data.dropna(inplace=True)  # Drop rows with missing values
    return data

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    print(data.head())