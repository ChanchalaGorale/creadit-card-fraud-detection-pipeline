import pandas as pd
import joblib
import json
import os
import logging
from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import mlflow
import mlflow.sklearn
from mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="data/logs/stage_04_train_baseline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_baseline_model(
    input_dir: str,
    output_dir: str
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a baseline logistic regression model and log metrics.

    Args:
        input_dir (str): Directory containing X_train.csv, y_train.csv, X_test.csv, y_test.csv.
        output_dir (str): Directory to save the trained model and metrics.

    Returns:
        Tuple[LogisticRegression, Dict[str, float]]: Trained model and evaluation metrics.
    """
    try:
        # Load data
        X_train = pd.read_csv(f"{input_dir}/X_train.csv")
        y_train = pd.read_csv(f"{input_dir}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{input_dir}/X_test.csv")
        y_test = pd.read_csv(f"{input_dir}/y_test.csv").values.ravel()

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="Baseline Model"):

            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="baseline_model")
            mlflow.log_params(model.get_params())

            os.makedirs(output_dir, exist_ok=True)

            # Save model
            joblib.dump(model, f"{output_dir}/model.pkl")

            # Save metrics
            with open(f"{output_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Baseline model trained. Metrics: {metrics}")
            return model, metrics

    except Exception as e:
        logging.error(f"Error training baseline model: {e}")
        raise
