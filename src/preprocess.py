import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # keep letters, spaces, apostrophes
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    tokens = word_tokenize(text)

    # keep all alphabetic/apostrophe tokens, including "i"
    filtered_tokens = [word for word in tokens if re.match(r"^[a-zA-Z']+$", word)]

    return " ".join(filtered_tokens)


def preprocess_text(text):
    cleaned_text = clean_text(text)
    processed_text = tokenize_text(cleaned_text)
    return processed_text