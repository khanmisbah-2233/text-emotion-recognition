import pandas as pd
from src.config import DATASET_FILE, TEXT_COLUMN, LABEL_COLUMN


def load_dataset():
    """
    Load dataset from CSV file and validate required columns.
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)

    if df.empty:
        raise ValueError("Dataset file is empty.")

    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    return df