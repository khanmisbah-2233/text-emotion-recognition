from src.config import METRICS_REPORT_FILE


def save_metrics_report(precision, recall, f1_score, classification_rep):
    """
    Save evaluation metrics and classification report to a text file.
    """
    report_text = f"""
===== MODEL EVALUATION REPORT =====

Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1_score:.4f}

----- Classification Report -----

{classification_rep}
"""

    with open(METRICS_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)