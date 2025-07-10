import logging
import time
import mlflow
from mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

from stages.stage_01_load_clean_data import load_data
from stages.stage_02_identify_feature_and_target import balanced_feature_identification
from stages.stage_03_split_data import split_and_save_data
from stages.stage_04_train_baseline import train_baseline_model
from stages.stage_05_find_best_model import find_best_model
from stages.stage_06_train_best_model import train_best_model


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logging.basicConfig(
    filename="data/logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_stage(stage_name: str, func, *args, **kwargs) -> None:
    """Log execution time and errors for each pipeline stage."""
    logging.info(f"Starting {stage_name}")
    start_time = time.time()
    try:
        func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(f"Completed {stage_name} in {duration:.2f} seconds")
    except Exception as e:
        logging.exception(f"Error in stage '{stage_name}': {e}")
        raise

def run_pipeline() -> None:
    """Run the end-to-end credit card fraud detection model training pipeline."""
    logging.info("Starting full fraud detection model pipeline")
    try:
        log_stage("Stage 01 - Load & Clean Data", load_data,
                  input_path="data/raw/data.csv",
                  output_dir="data/clean")

        log_stage("Stage 02 - Balance & Select Features", balanced_feature_identification,
                  input_path="data/clean/data.csv",
                  output_dir="data/split",
                  target="Class")

        log_stage("Stage 03 - Split Data", split_and_save_data,
                  input_dir="data/split",
                  output_dir="data/train_test")

        log_stage("Stage 04 - Train Baseline Model", train_baseline_model,
                  input_dir="data/train_test",
                  output_dir="models/baseline")

        log_stage("Stage 05 - Find Best Model & Params", find_best_model,
                  input_dir="data/train_test",
                  output_dir="models/best")

        log_stage("Stage 06 - Train Final Best Model", train_best_model,
                  input_dir="data/train_test",
                  best_dir="models/best",
                  output_dir="models/final")

        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.exception(f"Pipeline failed with error: {e}")

if __name__ == "__main__":
    run_pipeline()
