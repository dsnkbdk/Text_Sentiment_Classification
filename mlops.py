import mlflow
import logging
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.exceptions import RestException
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)

def relegate_and_promote(
    registered_name: str,
    candidate_version: str,
    candidate_f1_score: float
) -> None:

    # Initialize mlflow client
    client = MlflowClient()
    
    # Get the prod model
    try:
        prod_model = client.get_model_version_by_alias(registered_name, "production")
    except RestException:
        prod_model = None

    # Prod model does not exist
    if prod_model is None:
        # Promote to prod model
        client.set_registered_model_alias(registered_name, "production", candidate_version)
        logger.info(f"Prod model is None, candidate model (ver.{candidate_version}) is promoted")
    
    # Prod model already exists
    else:
        # Get the f1 score of the prod model
        prod_f1_score = client.get_run(prod_model.run_id).data.metrics.get("test_f1_score")
        
        # If the candidate model is better
        if prod_f1_score is None or prod_f1_score < candidate_f1_score:
            # Remove the current prod model
            client.delete_registered_model_alias(registered_name, "production")
            # Promote to prod model
            client.set_registered_model_alias(registered_name, "production", candidate_version)
            logger.info(f"Candidate model (ver.{candidate_version}) is better, prod model (ver.{prod_model.version}) is deprecated")
        
        # If the prod model is better
        else:
            logger.info(f"Prod model (ver.{prod_model.version}) is better, candidate model (ver.{candidate_version}) is deprecated")


def ml_workflow(
    *,
    experiment_name: str,
    run_name_prefix: str,
    classifier: callable,
    registered_model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    param_grid: dict,
    random_state: int
) -> None:
    
    # Initialize mlflow
    mlflow.set_experiment(experiment_name)

    # Autolog metrics and parameters
    mlflow.sklearn.autolog(log_models=False, silent=True)
    
    with mlflow.start_run(run_name=f"{run_name_prefix} {pd.Timestamp.now().floor('s')}"):

        # Fitting
        logger.info("Start fitting the model")

        clf = classifier(
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            random_state=random_state
        )
        
        logger.info("Fitting complete")
        
        best_estimator = clf.best_estimator_

        # Evaluate
        y_pred = best_estimator.predict(X_test)
        test_f1_score = f1_score(y_test, y_pred, average="macro")
        
        # Log the evaluation results
        mlflow.log_metric("test_f1_score", test_f1_score)
        mlflow.log_text(classification_report(y_test, y_pred), "test_report.txt")

        # Log and register the best model
        model_info = mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name="best_estimator",
            registered_model_name=registered_model_name,
            signature=infer_signature(X_test, y_pred)
        )
        logger.info(f"The best model (ver.{model_info.registered_model_version}) is registered")

        # Model promotion
        relegate_and_promote(
            registered_name=registered_model_name,
            candidate_version=model_info.registered_model_version,
            candidate_f1_score=test_f1_score
        )


def llm_workflow(
    *,
    experiment_name: str,
    run_name_prefix: str,
    classifier: callable,
    model: str,
    registered_model_name: str,
    X_test,
    y_test,
    batch_size: int,
    max_length: int,
    device: int | None = None
) -> None:
    
    # Initialize mlflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{run_name_prefix} {pd.Timestamp.now().floor('s')}"):
        
        # Log params
        mlflow.log_params(
            {
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "max_length": max_length
            }
        )

        clf = classifier(model=model, device=device)

        logger.info(f"Loading complete, start inferencing")
        
        # Inference
        pred = clf(
            inputs=X_test.to_list(),
            batch_size=batch_size,
            max_length=max_length,
            truncation=True
        )
        
        y_pred: list[str] = []
        y_score: list[float] = []
        
        for dict_p in pred:
            # {'label': 'negative/neutral/positive', 'score': float}
            y_pred.append(dict_p["label"])
            y_score.append(dict_p["score"])

        # Evaluate
        test_f1_score = f1_score(y_test, y_pred, average="macro")

        # Log the evaluation results
        mlflow.log_metric("test_f1_score", test_f1_score)
        mlflow.log_text(classification_report(y_test, y_pred), "test_report.txt")

        pip_requirements = [
            "huggingface_hub==1.3.4",
            "numpy==1.26.4",
            "packaging==25.0",
            "safetensors==0.7.0",
            "tokenizers==0.22.2",
            "torch==2.2.2",
            "transformers==5.0.0"
        ]

        # Log and register the best model
        model_info = mlflow.transformers.log_model(
            transformers_model=clf,
            name="best_estimator",
            registered_model_name=registered_model_name,
            signature=infer_signature(X_test.to_list(), y_pred),
            pip_requirements=pip_requirements
        )
        logger.info(f"The best model (ver.{model_info.registered_model_version}) is registered")
        
        # Model promotion
        relegate_and_promote(
            registered_name=registered_model_name,
            candidate_version=model_info.registered_model_version,
            candidate_f1_score=test_f1_score
        )




if __name__ == '__main__':

    from dotenv import load_dotenv
    from data import data_preparation
    from ml_model import logistic_regression
    from llm_model import cardiffnlp_roberta
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    load_dotenv()
    RANDOM_STATE = 2026
    ML_REGISTERED_MODEL_NAME = "TFIDF_Logistic_Regression"
    LLM_REGISTERED_MODEL_NAME = "HF_Cardiffnlp_RoBERTa_Sentiment"

    # DATA
    try:
        X_train, X_test, y_train, y_test = data_preparation(
            dataset="oliviervha/crypto-news",
            file_name="cryptonews.csv",
            random_state=RANDOM_STATE
        )
    except Exception:
        logger.exception("Smoke test failed")
        raise

    # ML
    # try:
    #     param_grid = {
    #         "tfidf__ngram_range": [(1, 1), (1, 2)],
    #         "tfidf__max_df": [0.8, 0.9],
    #         "tfidf__min_df": [5, 10],
    #         "clf__max_iter": [200, 500]
    #     }

    #     ml_workflow(
    #         experiment_name="Sentiment_Logistic_Regression",
    #         run_name_prefix="logreg_gridsearch",
    #         classifier=logistic_regression,
    #         registered_model_name=ML_REGISTERED_MODEL_NAME,
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         param_grid=param_grid,
    #         random_state=RANDOM_STATE
    #     )
    # except Exception:
    #     logger.exception("Smoke test failed")
    #     raise

    # LLM
    try:
        llm_workflow(
        experiment_name="Sentiment_Open_Source_LLM",
        run_name_prefix="cardiffnlp_pipeline",
        classifier=cardiffnlp_roberta,
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        registered_model_name=LLM_REGISTERED_MODEL_NAME,
        X_test=X_test,
        y_test=y_test,
        batch_size=16,
        max_length=256
    )
    except Exception:
        logger.exception("Smoke test failed")
        raise

