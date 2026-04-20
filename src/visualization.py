import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CONFUSION_MATRIX_FILE


def plot_confusion_matrix(confusion_mat, class_names):
    """
    Plot and save confusion matrix as heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()