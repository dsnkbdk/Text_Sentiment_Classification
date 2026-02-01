import os
import sys
import logging
from dotenv import load_dotenv

from data import data_preparation
from model import logistic_regression
from mlops import model_workflow

logger = logging.getLogger(__name__)

# Config
DATASET = "oliviervha/crypto-news"
FILE_NAME = "cryptonews.csv"
RANDOM_STATE = 2026

EXPERIMENT_NAME = "Sentiment_Logistic_Regression"
RUN_NAME_PREFIX = "logreg_gridsearch"
REGISTERED_MODEL_NAME = "TFIDF_Logistic_Regression"

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.8, 0.9],
    "tfidf__min_df": [5, 10],
    "clf__max_iter": [200, 500]
}

def config_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run_workflow() -> None:

    # Data
    X_train, X_test, y_train, y_test = data_preparation(
        dataset=DATASET,
        file_name=FILE_NAME,
        random_state=RANDOM_STATE
    )

    # Model
    model_workflow(
        experiment_name=EXPERIMENT_NAME,
        run_name_prefix=RUN_NAME_PREFIX,
        Classifier=logistic_regression,
        registered_model_name=REGISTERED_MODEL_NAME,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        param_grid=PARAM_GRID,
        random_state=RANDOM_STATE
    )

def main() -> int:
    
    load_dotenv()
    config_logging()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"MLFLOW_TRACKING_URI is {mlflow_tracking_uri}")

    try:
        run_workflow()
        logger.info("Workflow finished successfully")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Workflow failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())