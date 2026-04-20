import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

from src.config import MODEL_FILE, LABEL_ENCODER_FILE, TEST_DATA_FILE
from src.utils import load_object
from src.visualization import plot_confusion_matrix
from src.report import save_metrics_report


def evaluate_model():
    model = load_model(MODEL_FILE)
    label_encoder = load_object(LABEL_ENCODER_FILE)

    test_data = np.load(TEST_DATA_FILE)
    X_test_pad = test_data["X_test_pad"]
    y_test = test_data["y_test"]

    y_pred_prob = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, label_encoder.classes_)
    save_metrics_report(weighted_precision, weighted_recall, weighted_f1, report)

    print("Evaluation completed successfully.")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall   : {weighted_recall:.4f}")
    print(f"Weighted F1-score : {weighted_f1:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1-score    : {macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    evaluate_model()