import pandas as pd
import logging
from typing import Optional

logging.basicConfig(
    filename="data/logs/stage_01_load_clean_data.log",
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(input_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Loads a CSV file and optionally saves it to another path.

    Args:
        file_path (str): Path to the input CSV file.
        save_path (Optional[str]): If provided, saves the loaded DataFrame to this path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data from {input_path} with shape {df.shape}")

        if df.isnull().sum().any():
            logging.warning(f"Found {df.isnull().sum().sum()} missing values. Filling with mode per column.")
            
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
                    else:
                        logging.warning(f"Column '{col}' has nulls but no mode could be computed.")
        else:
            logging.info("No missing values found.")

        if output_dir:
            df.to_csv(f"{output_dir}/data.csv", index=False)
            logging.info(f"Saved loaded data to {output_dir}")

        return df

    except FileNotFoundError:
        logging.error(f"File not found: {output_dir}")
        raise

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
