import pandas as pd
import logging
from typing import Tuple
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    filename="data/logs/stage_02_identify_feature_and_target.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def balanced_feature_identification(input_path: str, output_dir: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load cleaned data, handle class imbalance, and save processed outputs.

    Args:
        input_path (str): Path to the cleaned CSV file.
        output_dir (str): Directory to save the processed feature and target files.
        target (str): Name of the target column in the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target labels.
    """
    try:
        df = pd.read_csv(input_path)

        class_counts = df[target].value_counts(normalize=True)
        fraud_percent = class_counts.get(1, 0) * 100

        X = df.drop(columns=[target])
        y = df[target]

        if fraud_percent < 5:
            logging.info("Class imbalance found: running SMOTE for resampling.")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y.values.ravel())

            X = pd.DataFrame(X_resampled, columns=X.columns)
            y = pd.Series(y_resampled, name=target)
        else:
            logging.info("No class imbalance found, skipping SMOTE.")


        X.to_csv(f"{output_dir}/X_data.csv", index=False)
        y.to_csv(f"{output_dir}/y_data.csv", index=False)

        logging.info("Preprocessing completed: features & targets separated and saved.")
        return X, y

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise
