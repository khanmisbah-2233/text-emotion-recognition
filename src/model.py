from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout,
    SpatialDropout1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.optimizers import Adam

from src.config import RNN_UNITS, DROPOUT_RATE, LEARNING_RATE


def build_rnn_model(vocab_size, embedding_dim, embedding_matrix, num_classes):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True
        ),
        SpatialDropout1D(0.25),
        SimpleRNN(
            RNN_UNITS,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model