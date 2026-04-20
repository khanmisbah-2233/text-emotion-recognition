import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    MODEL_FILE,
    TOKENIZER_FILE,
    LABEL_ENCODER_FILE,
    MAX_SEQUENCE_LENGTH,
    UNCLEAR_THRESHOLD,
)
from src.preprocess import preprocess_text
from src.utils import load_object


HAPPINESS_WORDS = {
    "happy", "joy", "joyful", "glad", "delighted", "cheerful", "smile",
    "smiling", "best", "great", "amazing", "wonderful", "awesome",
    "fantastic", "beautiful", "perfect"
}

SADNESS_WORDS = {
    "sad", "lonely", "broken", "cry", "crying", "depressed", "hurt",
    "pain", "upset", "lost", "hopeless", "miserable", "heartbroken",
    "sorrow", "grief"
}

HATE_WORDS = {
    "hate", "despise", "disgust", "disgusted", "loath", "detest"
}

ANGER_WORDS = {
    "angry", "mad", "furious", "annoyed", "rage", "irritated", "frustrated"
}

LOVE_WORDS = {
    "love", "loved", "loving", "adore", "adorable", "romantic", "dear",
    "sweetheart", "beloved", "affection"
}

WORRY_WORDS = {
    "worried", "worry", "anxious", "nervous", "afraid", "fear", "fearful",
    "scared", "tense", "uneasy", "concerned"
}

RELIEF_WORDS = {
    "relieved", "relief", "finally", "thank god", "thankfully", "safe now",
    "stress is gone", "pressure is gone", "at ease"
}

FUN_WORDS = {
    "fun", "funny", "laugh", "laughed", "laughing", "playful", "enjoyed",
    "entertaining", "hilarious"
}

EMPTY_WORDS = {
    "empty", "numb", "blank", "nothing inside", "void", "hollow", "lifeless"
}

ENTHUSIASM_WORDS = {
    "excited", "enthusiastic", "motivated", "eager", "energetic", "inspired",
    "cannot wait", "can't wait", "looking forward"
}

SURPRISE_WORDS = {
    "surprised", "shock", "shocked", "wow", "unexpected", "suddenly",
    "unbelievable", "astonished", "amazed"
}

BOREDOM_WORDS = {
    "bored", "boring", "tired of", "nothing interesting", "dull", "routine",
    "monotonous"
}

NEUTRAL_WORDS = {
    "meeting", "market", "office", "schedule", "tomorrow", "today", "went",
    "starts", "report", "class", "work", "home"
}

SURPRISE_PHRASES = {
    "cannot believe",
    "can't believe",
    "did not expect",
    "didn't expect",
    "what just happened"
}


def keyword_fallback(raw_text: str):
    text = raw_text.lower()
    words = set(re.findall(r"[a-zA-Z']+", text))

    # phrase checks first
    for phrase in SURPRISE_PHRASES:
        if phrase in text:
            return "surprise"

    if words & HATE_WORDS:
        return "hate"
    if words & ANGER_WORDS:
        return "anger"
    if words & SADNESS_WORDS:
        return "sadness"
    if words & WORRY_WORDS:
        return "worry"
    if words & LOVE_WORDS:
        return "love"
    if words & RELIEF_WORDS:
        return "relief"
    if words & FUN_WORDS:
        return "fun"
    if words & EMPTY_WORDS:
        return "empty"
    if words & ENTHUSIASM_WORDS:
        return "enthusiasm"
    if words & SURPRISE_WORDS:
        return "surprise"
    if words & BOREDOM_WORDS:
        return "boredom"
    if words & HAPPINESS_WORDS:
        return "happiness"

    # neutral only as weak fallback
    if words & NEUTRAL_WORDS:
        return "neutral"

    return None


def predict_emotion(text):
    model = load_model(MODEL_FILE)
    tokenizer = load_object(TOKENIZER_FILE)
    label_encoder = load_object(LABEL_ENCODER_FILE)

    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction_probs = model.predict(padded_sequence, verbose=0)[0]
    predicted_index = int(np.argmax(prediction_probs))
    confidence_score = float(np.max(prediction_probs))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    sorted_indices = np.argsort(prediction_probs)[::-1][:3]
    top3_labels = label_encoder.inverse_transform(sorted_indices)
    top3_scores = [float(prediction_probs[i]) for i in sorted_indices]
    top_predictions = list(zip(top3_labels, top3_scores))

    message = ""

    fallback_label = keyword_fallback(text)

    # if model says neutral but strong cue exists, override
    if predicted_label == "neutral" and fallback_label is not None and fallback_label != "neutral":
        predicted_label = fallback_label
        message = f"Model leaned neutral, so keyword-based fallback selected '{fallback_label}'."

    elif confidence_score < UNCLEAR_THRESHOLD:
        message = "Low confidence. Emotion may be unclear or mixed."

    return predicted_label, confidence_score, message, top_predictions