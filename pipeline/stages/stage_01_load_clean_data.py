import pandas as pd
import logging

logging.basicConfig(
    filename="data/logs/stage_01_load_clean_data.log",
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(input_path: str, realtime_input_path: str, output_dir: str, retrain:bool) -> pd.DataFrame:
    """
    Loads a CSV file and optionally merges with a realtime CSV file, cleans missing values, and saves to output directory.

    Args:
        input_path (str): Path to the main input CSV file.
        realtime_input_path (str): Path to the realtime input CSV file (used if retrain is True).
        output_dir (str): Directory where the cleaned data will be saved.
        retrain (bool): If True, merges realtime data with main data.

    Returns:
        pd.DataFrame: The loaded and cleaned DataFrame.
    """
    try:
        df = pd.read_csv(input_path)

        if retrain and realtime_input_path:
            realtime_df = pd.read_csv(realtime_input_path)
            
            # Ensure both dataframes have the same columns by union of columns
            all_columns = df.columns.union(realtime_df.columns)

            df = df.reindex(columns=all_columns)
            realtime_df = realtime_df.reindex(columns=all_columns)

            # Vertically concatenate (stack)
            df = pd.concat([df, realtime_df], ignore_index=True, sort=False)

            # move realtime data to raw data
            df.to_csv(input_path, index=False)
            pd.DataFrame(columns=df.columns).to_csv(realtime_input_path, index=False)


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
