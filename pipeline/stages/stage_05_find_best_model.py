import pandas as pd
import joblib
import json
import os
import logging
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

logging.basicConfig(
    filename="data/logs/stage_05_find_best_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

def find_best_model(
    input_dir: str,
    output_dir: str
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Finds and trains the best classification model using hyperparameter tuning, evaluates it, and saves the results.
    
    Args:
        input_dir (str): Path to the directory containing the training and test CSV files (`X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`).
        output_dir (str): Directory where the best model, metrics, and parameters will be saved.

    Returns:
        Tuple[Any, Dict[str, Any], Dict[str, float]]:
            - The best trained model object.
            - The best hyperparameters as a dictionary.
            - The evaluation metrics (including F1 score) as a dictionary.
    """
    try:
        X_train = pd.read_csv(f"{input_dir}/X_train.csv")
        y_train = pd.read_csv(f"{input_dir}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{input_dir}/X_test.csv")
        y_test = pd.read_csv(f"{input_dir}/y_test.csv").values.ravel()

        os.makedirs(output_dir, exist_ok=True)

        # Define models and hyperparameter grids
        models_params = {
            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            ),
            "XGBoost": (
                XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.1, 0.01]
                }
            )
        }

        best_model = None
        best_model_name = ""
        best_params = {}
        best_metrics = {"f1_score": 0.0}

        for name, (model, param_grid) in models_params.items():
            logging.info(f"Tuning {name}...")
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test)

            logging.info(f"{name} | F1: {metrics['f1_score']} | Params: {grid_search.best_params_}")

            if metrics["f1_score"] > best_metrics["f1_score"]:
                best_model = grid_search.best_estimator_
                best_model_name = name
                best_params = grid_search.best_params_
                best_metrics = metrics

        # Save best model, metrics, and params
        joblib.dump(best_model, f"{output_dir}/model.pkl")

        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=4)

        with open(f"{output_dir}/params.json", "w") as f:
            json.dump({"model": best_model_name, "params": best_params}, f, indent=4)

        logging.info(f"Best Model: {best_model_name} | F1: {best_metrics['f1_score']}")
        return best_model, best_params, best_metrics

    except Exception as e:
        logging.error(f"Error in model refinement: {e}")
        raise
