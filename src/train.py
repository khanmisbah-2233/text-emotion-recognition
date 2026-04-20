import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MAX_WORDS,
    MAX_SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    MODEL_FILE,
    TOKENIZER_FILE,
    LABEL_ENCODER_FILE,
    TEST_DATA_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    EMBEDDING_DIM,
    WORD2VEC_WINDOW,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_WORKERS,
)
from src.data_loader import load_dataset
from src.preprocess import preprocess_text
from src.model import build_rnn_model
from src.utils import create_directories, encode_labels, save_object


def build_embedding_matrix(tokenizer, word2vec_model, max_words, embedding_dim):
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    embedding_matrix = np.random.normal(scale=0.05, size=(vocab_size, embedding_dim))

    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue
        if word in word2vec_model.wv:
            embedding_matrix[index] = word2vec_model.wv[word]

    return embedding_matrix, vocab_size


def rebalance_training_dataframe(df, label_column, random_state=42):
    """
    Balanced-ish training set:
    - very large classes -> cap at 5000
    - medium classes -> target 3000
    - very small classes -> target 1000
    """
    counts = df[label_column].value_counts()
    balanced_parts = []

    for label, count in counts.items():
        class_df = df[df[label_column] == label]

        if count >= 5000:
            target = 5000
            sampled = class_df.sample(n=target, random_state=random_state)

        elif count >= 1000:
            target = 3000
            sampled = class_df.sample(
                n=target,
                replace=(count < target),
                random_state=random_state
            )

        else:
            target = 1000
            sampled = class_df.sample(
                n=target,
                replace=True,
                random_state=random_state
            )

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts, axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df


def train_model():
    create_directories()

    df = load_dataset()
    df["clean_text"] = df[TEXT_COLUMN].apply(preprocess_text)

# Remove exact duplicate text-label pairs
    df = df.drop_duplicates(subset=["clean_text", LABEL_COLUMN]).reset_index(drop=True)

    # Original split first
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN]
    )

    # Rebalance only training data
    train_df = rebalance_training_dataframe(
        train_df,
        label_column=LABEL_COLUMN,
        random_state=RANDOM_STATE
    )

    print("Training class distribution after rebalancing:")
    print(train_df[LABEL_COLUMN].value_counts())

    print("\nTest class distribution:")
    print(test_df[LABEL_COLUMN].value_counts())

    # Encoder on full label space
    label_encoder, _ = encode_labels(df[LABEL_COLUMN])

    y_train = label_encoder.transform(train_df[LABEL_COLUMN])
    y_test = label_encoder.transform(test_df[LABEL_COLUMN])

    X_train = train_df["clean_text"]
    X_test = test_df["clean_text"]

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    train_tokens = [text.split() for text in X_train]

    word2vec_model = Word2Vec(
        sentences=train_tokens,
        vector_size=EMBEDDING_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS
    )

    embedding_matrix, vocab_size = build_embedding_matrix(
        tokenizer=tokenizer,
        word2vec_model=word2vec_model,
        max_words=MAX_WORDS,
        embedding_dim=EMBEDDING_DIM
    )

    num_classes = len(label_encoder.classes_)

    model = build_rnn_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        num_classes=num_classes
    )

    callbacks = [
       
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    save_object(tokenizer, TOKENIZER_FILE)
    save_object(label_encoder, LABEL_ENCODER_FILE)

    np.savez(TEST_DATA_FILE, X_test_pad=X_test_pad, y_test=y_test)

    return {
        "model": model,
        "history": history
    }


if __name__ == "__main__":
    train_model()