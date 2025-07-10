import pandas as pd
import os
import logging
from typing import Optional

logging.basicConfig(
    filename="data/logs/io_utils.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_csv(file_path: str, required_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Loads a CSV file and performs basic validation.

    Args:
        file_path (str): Path to the CSV file.
        required_columns (Optional[list[str]]): List of columns that must exist.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            logging.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

    logging.info(f"Loaded data from {file_path} with shape {df.shape}")
    return df
