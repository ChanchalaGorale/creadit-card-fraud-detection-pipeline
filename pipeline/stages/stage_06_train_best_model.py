import pandas as pd
import joblib
import json
import os
import logging
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="data/logs/stage_06_train_best_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def diagnose_model_fit(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> str:
    """Diagnose whether the model is overfitting, underfitting, or generalizing well."""
    threshold_gap = 0.1  # 10% difference considered significant
    low_metric_threshold = 0.65  # Define what we call "low" performance (F1, Recall, etc.)

    train_f1 = train_metrics.get("f1_score", 0)
    test_f1 = test_metrics.get("f1_score", 0)
    train_recall = train_metrics.get("recall", 0)
    test_recall = test_metrics.get("recall", 0)

    f1_gap = train_f1 - test_f1
    recall_gap = train_recall - test_recall

    # Check for underfitting
    if train_f1 < low_metric_threshold and test_f1 < low_metric_threshold:
        return "🔴 Model is underfitting: both train and test F1 scores are low."

    # Check for overfitting
    elif f1_gap > threshold_gap or recall_gap > threshold_gap:
        return "🟠 Model is overfitting: large gap between train and test performance."

    # Model is generalizing
    else:
        return "🟢 Model is generalizing well: train and test metrics are consistent."


def train_best_model(
    input_dir: str,
    best_dir: str,
    output_dir: str
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """
    Train the final production model on the full balanced dataset using the best hyperparameters.

    Args:
        input_dir (str): Directory containing X_data.csv and y_data.csv.
        best_dir (str): Directory containing params.json with best hyperparameters.
        output_dir (str): Directory to save the trained model and metrics.

    Returns:
        Tuple[XGBClassifier, Dict[str, float]]: Trained model and evaluation metrics.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        X_train = pd.read_csv(f"{input_dir}/X_train.csv")
        y_train = pd.read_csv(f"{input_dir}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{input_dir}/X_test.csv")
        y_test = pd.read_csv(f"{input_dir}/y_test.csv").values.ravel()

        best_params_path = os.path.join(best_dir, "params.json")

        if not os.path.exists(best_params_path):
            raise FileNotFoundError(f"{best_params_path} does not exist.")

        with open(best_params_path, "r") as f:
            data = json.load(f)
        
        best_model_name = data["model"]
        best_params = data["params"]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="Final Best Model"):

            # Train model
            if best_model_name == "RandomForest":
                model = RandomForestClassifier(**best_params)
            else:
                # when best model is XGBoost or if no best model found 
                model = XGBClassifier(**best_params)
            
            model.fit(X_train, y_train)

            # Inference input and output (for signature)
            input_example = X_train.iloc[:1]  # one row
            prediction_example = model.predict(input_example)

            # Infer the signature (schema of inputs/outputs)
            signature = infer_signature(input_example, prediction_example)

            mlflow.log_params(best_params)
            mlflow.xgboost.log_model(model, name="final_model", input_example=input_example, signature=signature)

            # Predict and evaluate using threshold
            y_probs_train = model.predict_proba(X_train)[:, 1]
            y_pred_train = (y_probs_train >= 0.5).astype(int)

            y_probs_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_probs_test >= 0.5).astype(int)

            train_metrics = {
                "accuracy": accuracy_score(y_train, y_pred_train),
                "precision": precision_score(y_train, y_pred_train),
                "recall": recall_score(y_train, y_pred_train),
                "f1_score": f1_score(y_train, y_pred_train)
            }

            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test),
                "recall": recall_score(y_test, y_pred_test),
                "f1_score": f1_score(y_test, y_pred_test)
            }
            
            result = diagnose_model_fit(train_metrics, test_metrics)

            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.set_tag("Model performance: ", result)

            logging.info(f"Final Model Performance: {result}")
            logging.info(f"Final Model Metrics: {test_metrics}")

            # Save model and metrics
            joblib.dump(model, os.path.join(output_dir, "model.pkl"))

            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=4)

            with open(os.path.join(output_dir, "threshold.json"), "w") as f:
                json.dump({"threshold": 0.5}, f, indent=4)

            return model, test_metrics

    except Exception as e:
        logging.error(f"Error in final model training: {e}")
        raise
