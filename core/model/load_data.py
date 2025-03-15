import pandas as pd

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
    data["label"] = data["label"].astype(int)  # Convert labels to integers
    data.dropna(inplace=True)  # Drop rows with missing values
    return data

if __name__ == "__main__":
    # Example usage
    data = load_data("data/star_data.txt")
    data = preprocess_data(data)
    print(data.head())