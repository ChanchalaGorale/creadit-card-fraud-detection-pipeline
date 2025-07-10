import pandas as pd
import os
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename="data/logs/stage_03_split_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def split_and_save_data(
    input_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets and save them.

    Args:
        input_dir (str): Directory containing 'X_data.csv' and 'y_data.csv'.
        output_dir (str): Directory to save train/test splits.
        test_size (float): Fraction of data to use as test.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    try:
        X = pd.read_csv(f"{input_dir}/X_data.csv")
        y = pd.read_csv(f"{input_dir}/y_data.csv")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        logging.info(f"Data split: {len(X_train)} train / {len(X_test)} test samples.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error during data split: {e}")
        raise
