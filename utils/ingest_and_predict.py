import os
import pandas as pd
import joblib
from typing import Dict

REALTIME_CSV = "data/realtime/data.csv"
MODEL_PATH = "models/final/model.pkl"
FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20","V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]

def get_next_time_value(csv_path: str) -> float:
    if not os.path.exists(csv_path):
        return 0.0
    
    df = pd.read_csv(csv_path)

    if df.empty or "Time" not in df.columns:
        return 0.0

    max_time = df["Time"].max()
    if pd.isna(max_time):
        return 0.0

    return float(max_time) + 1.0

async def predict_and_store(input_data: Dict[str, float]) -> int:
    """
    Predict fraud and append input + prediction to data/realtime/data.csv
    """
    print("train data\n",pd.read_csv("data/train_test/X_train.csv").columns)
    # Convert input dict to DataFrame
    input_data["Time"] = get_next_time_value("data/realtime/data.csv")
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_COLUMNS]

    print("input\n",input_df.columns)

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Make prediction
    prediction = model.predict(input_df)[0]
    input_df["Class"] = prediction

    # Create realtime folder if it doesn't exist
    os.makedirs(os.path.dirname(REALTIME_CSV), exist_ok=True)

    # Append to CSV
    if os.path.exists(REALTIME_CSV):
        input_df.to_csv(REALTIME_CSV, mode='a', header=False, index=False)
    else:
        input_df.to_csv(REALTIME_CSV, mode='w', header=True, index=False)

    return prediction