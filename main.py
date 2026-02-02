import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from data import data_preparation
from ml_model import logistic_regression
from llm_model import cardiffnlp_roberta
from mlops import ml_workflow, llm_workflow

logger = logging.getLogger(__name__)

# Config
DATASET = "oliviervha/crypto-news"
FILE_NAME = "cryptonews.csv"
RANDOM_STATE = 2026

# ML
ML_EXPERIMENT_NAME = "Sentiment_Logistic_Regression"
ML_RUN_NAME_PREFIX = "logreg_gridsearch"
ML_REGISTERED_MODEL_NAME = "TFIDF_Logistic_Regression"

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.8, 0.9],
    "tfidf__min_df": [5, 10],
    "clf__max_iter": [200, 500]
}

# LLM
LLM_EXPERIMENT_NAME = "Sentiment_Open_Source_LLM"
LLM_RUN_NAME_PREFIX = "cardiffnlp_pipeline"
LLM_REGISTERED_MODEL_NAME = "HF_Cardiffnlp_RoBERTa_Sentiment"
LLM_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LLM_BATCH_SIZE = 16
LLM_MAX_LENGTH = 256


def config_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training pipeline for sentiment classification"
    )

    # Mode switch
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-ml",
        action="store_true",
        help="Run traditional ML workflow"
    )
    mode.add_argument(
        "-llm",
        action="store_true",
        help="Run open-source LLM workflow"
    )

    # Shared
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    
    # ML args
    parser.add_argument("--ml-experiment-name", type=str, default=ML_EXPERIMENT_NAME)
    parser.add_argument("--ml-run-name-prefix", type=str, default=ML_RUN_NAME_PREFIX)
    parser.add_argument("--ml-registered-model-name", type=str, default=ML_REGISTERED_MODEL_NAME)

    # LLM args
    parser.add_argument("--llm-experiment-name", type=str, default=LLM_EXPERIMENT_NAME)
    parser.add_argument("--llm-run-name-prefix", type=str, default=LLM_RUN_NAME_PREFIX)
    parser.add_argument("--llm-registered-model-name", type=str, default=LLM_REGISTERED_MODEL_NAME)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL)

    return parser.parse_args()


def run_ml_workflow(
    *,
    random_state: int,
    experiment_name: str,
    run_name_prefix: str,
    registered_model_name: str
) -> None:

    # Data
    X_train, X_test, y_train, y_test = data_preparation(
        dataset=DATASET,
        file_name=FILE_NAME,
        random_state=random_state
    )

    # ML
    ml_workflow(
        experiment_name=experiment_name,
        run_name_prefix=run_name_prefix,
        classifier=logistic_regression,
        registered_model_name=registered_model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        param_grid=PARAM_GRID,
        random_state=random_state
    )


def run_llm_workflow(
    *,
    random_state: int,
    experiment_name: str,
    run_name_prefix: str,
    registered_model_name: str,
    model: str,
    batch_size: int,
    max_length: int,
) -> None:

    # Data
    _, X_test, _, y_test = data_preparation(
        dataset=DATASET,
        file_name=FILE_NAME,
        random_state=random_state
    )

    # LLM
    llm_workflow(
        experiment_name=experiment_name,
        run_name_prefix=run_name_prefix,
        classifier=cardiffnlp_roberta,
        model=model,
        registered_model_name=registered_model_name,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        max_length=max_length
    )


def main() -> int:
    
    load_dotenv()
    config_logging()

    args = parse_args()

    run_ml_mode = True
    if args.llm:
        run_ml_mode = False
    if args.ml:
        run_ml_mode = True

    logger.info(f"Running {'ML' if run_ml_mode else 'LLM'} mode")
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"MLFLOW_TRACKING_URI is {mlflow_tracking_uri}")

    try:
        if run_ml_mode:
            logger.info(f"""This experiment is running ML with:
random_state = {args.random_state}
experiment_name = {args.ml_experiment_name}
run_name_prefix = {args.ml_run_name_prefix}
registered_model_name = {args.ml_registered_model_name}
"""
            )
            
            run_ml_workflow(
                random_state=args.random_state,
                experiment_name=args.ml_experiment_name,
                run_name_prefix=args.ml_run_name_prefix,
                registered_model_name=args.ml_registered_model_name
            )
        
        else:
            logger.info(f"""This experiment is running LLM with:
random_state = {args.random_state}
experiment_name = {args.llm_experiment_name}
run_name_prefix = {args.llm_run_name_prefix}
registered_model_name = {args.llm_registered_model_name}
model = {args.llm_model}
"""
            )
            
            run_llm_workflow(
                random_state=args.random_state,
                experiment_name=args.llm_experiment_name,
                run_name_prefix=args.llm_run_name_prefix,
                registered_model_name=args.llm_registered_model_name,
                model=args.llm_model,
                batch_size=LLM_BATCH_SIZE,
                max_length=LLM_MAX_LENGTH
            )

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