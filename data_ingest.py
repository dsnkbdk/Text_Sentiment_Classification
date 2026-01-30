import warnings
warnings.filterwarnings('ignore')

import json
import logging
import kagglehub
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def data_ingestion(dataset: str) -> pd.DataFrame:
    # Load .env
    load_dotenv()
    
    # Download dataset
    path = Path(kagglehub.dataset_download(dataset))
    files = list(path.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError(f"No CSV file found in path: {path}")
    
    if len(files) > 1:
        raise ValueError(f"Found {len(files)} CSV files in path: {path}")

    return pd.read_csv(files[0])


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
   
    # Drop any rows with any NA
    missing_values = df.isna().any(axis=1)
    
    if missing_values.any():
        logger.warning(f"Dropping {missing_values.sum()} rows due to missing values")
        df = df.loc[~missing_values].reset_index(drop=True)

    # Extract class
    df["class"] = df["sentiment"].apply(lambda x: json.loads(x.replace("'", '"')).get("class"))

    # Drop invalid class
    invalid_class = df["class"].isna() | ~df["class"].isin({"negative", "neutral", "positive"})

    if invalid_class.any():
        logger.warning(f"Dropping {invalid_class.sum()} rows due to invalid class")
        df = df.loc[~invalid_class].reset_index(drop=True)

    logger.info(f"Class distribution (count):\n{df["class"].value_counts().to_string()}")
    logger.info(f"Class distribution (ratio):\n{df["class"].value_counts(normalize=True).to_string()}")

    # Build title + text feature
    df["title_text"] = df["title"] + ". " + df["text"]

    return df


def data_preparation(dataset: str, test_size: float=0.25) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    df = data_cleaning(data_ingestion(dataset))
    X, y = df["title_text"], df["class"]

    return train_test_split(X, y, test_size=test_size, stratify=y)

