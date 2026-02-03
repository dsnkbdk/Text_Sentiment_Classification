import ast
import logging
import kagglehub
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def data_ingestion(dataset: str, file_name: str) -> pd.DataFrame:
    
    # Download dataset
    path = Path(kagglehub.dataset_download(dataset))
    file = path / file_name
    
    if not file.exists():
        raise FileNotFoundError(f"File '{file_name}' not found in path: {path}")
    
    df = pd.read_csv(file)
    
    logger.info(f"File '{file_name}' has been loaded with shape {df.shape}")
    
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Mandatory columns
    missing_mandatory = df[["date", "sentiment", "source", "subject"]].isna().any(axis=1)
    
    # text & title
    missing_text_title = df["text"].isna() & df["title"].isna()
    
    # Drop invalid rows
    invalid_rows = missing_mandatory | missing_text_title
    
    if invalid_rows.any():
        logger.warning(f"Dropping {invalid_rows.sum()} rows due to missing values")
        df = df.loc[~invalid_rows].reset_index(drop=True)

    # Extract sentiment field
    def parse_sentiment(x):
        if not isinstance(x, str):
            return {}
        try:
            field = ast.literal_eval(x)
            return field if isinstance(field, dict) else {}
        except Exception:
            return {}
    
    sentiment = df["sentiment"].apply(parse_sentiment)

    df["class"] = sentiment.apply(lambda x: x.get("class"))
    df["polarity"] = pd.to_numeric(sentiment.apply(lambda x: x.get("polarity")), errors="coerce")
    df["subjectivity"] = pd.to_numeric(sentiment.apply(lambda x: x.get("subjectivity")), errors="coerce")

    # Drop invalid class
    invalid_class = df["class"].isna() | ~df["class"].isin({"negative", "neutral", "positive"})

    if invalid_class.any():
        logger.warning(f"Dropping {invalid_class.sum()} rows due to invalid class")
        df = df.loc[~invalid_class].reset_index(drop=True)

    df = df.sort_values("date").reset_index(drop=True)

    # Class distribution
    logger.info(f"Class distribution (count):\n{df['class'].value_counts().to_string()}")
    logger.info(f"Class distribution (ratio):\n{df['class'].value_counts(normalize=True).to_string()}")

    # Build title + text feature
    df["input_text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.lstrip(". ")

    return df


def get_clean_data(dataset: str, file_name: str) -> pd.DataFrame:
    return data_cleaning(data_ingestion(dataset, file_name))


def data_preparation(
    dataset: str,
    file_name: str,
    random_state: int,
    test_size: float=0.25
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    df = get_clean_data(dataset, file_name)
    X, y = df["input_text"], df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Data preparation is complete")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    from dotenv import load_dotenv
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    load_dotenv()
    RANDOM_STATE = 2026
    
    try:
        X_train, X_test, y_train, y_test = data_preparation(
            dataset="oliviervha/crypto-news",
            file_name="cryptonews.csv",
            random_state=RANDOM_STATE
        )
    except Exception:
        logger.exception("Smoke test failed")
        raise