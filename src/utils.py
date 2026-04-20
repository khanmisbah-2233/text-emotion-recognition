import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def create_directories():
    """
    Ensure required directories exist.
    """
    folders = ["models", "outputs", "outputs/plots", "docs"]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)


def encode_labels(labels):
    """
    Encode text labels into numeric labels.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoder, encoded_labels


def save_object(obj, file_path):
    """
    Save Python object using joblib.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Load Python object using joblib.
    """
    return joblib.load(file_path)