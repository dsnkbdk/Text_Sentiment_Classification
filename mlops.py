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


def model_workflow(
    *,
    experiment_name: str,
    run_name_prefix: str,
    Classifier: callable,
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

        clf = Classifier(
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