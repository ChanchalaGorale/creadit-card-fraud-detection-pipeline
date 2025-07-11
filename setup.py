import os
import pandas as pd

def ensure_folders_exist(folders: list):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    raw_data = os.path.join("data", "raw", "data.csv")
    if not os.path.exists(raw_data):
        print(f"Raw data not found at {raw_data}")

def main():
    folders = [
        "data/raw", "data/clean", "data/split", "data/train_test", "data/logs","data/realtime",
        "models/baseline", "models/best", "models/final", "config"
    ]
    
    ensure_folders_exist(folders)

if __name__ == "__main__":
    main()
