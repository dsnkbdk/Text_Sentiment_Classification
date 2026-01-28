import warnings
warnings.filterwarnings('ignore')

import os
import mlflow
import logging
import numpy as np
import pandas as pd
import custom_functions as cf
import matplotlib.pyplot as plt

from darts import TimeSeries
from itertools import product
from mlflow import MlflowClient
from darts.models import XGBModel
from ingest import WindModelV4Data
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from mlflow.models import infer_signature

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')

# gauss engine
gauss_eng = create_engine(f"postgresql+psycopg2://wennan@trading-prod.cuvldknbzcxv.ap-southeast-2.rds.amazonaws.com:5432/gauss")

# mlflow engine
mlflow_eng = create_engine(f"postgresql+psycopg2://wennan@trading-prod.cuvldknbzcxv.ap-southeast-2.rds.amazonaws.com:5432/mlflow")


def get_data_for_train(
        ingestor: WindModelV4Data,
        params: dict,
        model_collection: str
) -> dict[str, dict[str, dict[str, TimeSeries]]]:
    """Get structured training data based on input parameters.

    Prepare data for model training using the `WindModelV4Data` ingestor.

    Args:
        ingestor:
            Instance of the `WindModelV4Data` class.
        params:
            A dict containing parameters `lags_type`, `tesla_lags`, `h` for customizing training data.
        model_collection:
            Model name.

    Returns:
        A nested dictionary of time series data for training, validation, and testing.
    """
    logger.info(f"{model_collection}:Getting data for training")

    # Get data for train
    train_data = ingestor.data_integration_output(
        lags_type=params['lags_type'],
        tesla_lags=params['tesla_lags'],
        h=params['h']
    )

    return train_data


def global_xgboost(
        train_data: dict,
        params: dict,
        hyperparams: dict,
        model_collection: str
) -> None:
    """Train a global XGBoost model using the Darts framework.

    Performs complete training workflow including: training, validation, and testing,
    logging parameters, historical forecasts, error metrics, plots to `MLflow`.

    Args:
        train_data: Structured training data from `get_data_for_train`, dictionary format.
        params: Model structure parameters, dictionary format.
        hyperparams: Model hyperparameters, dictionary format.
        model_collection: Model name, used for model identification, registration and query.

    Returns:
        None

    Raises:
        ValueError: If input lag type is incompatible with series length.
    """
    logger.info(
        f"{model_collection}:Start training with parameters\n"
        f"{[params['lags'], params['tesla_lags'], params['h'], params['window'], params['lags_type'], params['model_type']]} "
        f"{[hyperparams['n_estimators'], hyperparams['learning_rate']]} "
        f"{[hyperparams['max_depth'], hyperparams['reg_alpha'], hyperparams['reg_lambda']]}"
    )

    # Log parameters
    mlflow.log_params(params)
    mlflow.log_params(hyperparams)
    logger.info(f"{model_collection}:Parameters logging completed")

    # Unpack data
    future_covariates = [station_data['future_covariates'] for station_data in train_data['data'].values()]
    target_train = [station_data['target_train'] for station_data in train_data['data'].values()]
    target_valid = [station_data['target_valid'] for station_data in train_data['data'].values()]
    target_test = [station_data['target_test'] for station_data in train_data['data'].values()]
    station_list = list(train_data['data'].keys())
    lags_dict = train_data['lags_dict']

    # The length of (lags + output chunk) must be <= the length of target_valid or target_test
    if isinstance(params['lags'], int):
        req_length = params['lags'] + params['h']
    elif isinstance(params['lags'], list):
        req_length = abs(min(params['lags'])) + params['h']
    elif isinstance(params['lags'], dict):
        req_length = max(params['lags'].values()) + params['h']
    else:
        message = f"{model_collection}:Lags type must be integer, list or dict"
        logger.error(msg=message, exc_info=True)
        raise ValueError(message)

    min_valid_len = min(len(ts) for ts in target_valid)
    min_test_len = min(len(ts) for ts in target_test)

    if req_length > min_valid_len or req_length > min_test_len:
        message = f"{model_collection}:Required length ({req_length}) exceeds the length of validation series ({min_valid_len}) or test series ({min_test_len})"
        logger.error(msg=message, exc_info=True)
        raise ValueError(message)

    # Fit model
    logger.info(f"{model_collection}:Start fitting the model")

    model = XGBModel(
        lags=params['lags'],
        output_chunk_length=params['h'],
        lags_future_covariates=lags_dict,
        callbacks=[cf.CustomEvaluationMonitor(params=params, hyperparams=hyperparams, period=50)],
        **hyperparams
    )

    model.fit(
        series=[ts['scaled_mw'] for ts in target_train],
        future_covariates=future_covariates,
        val_series=[ts['scaled_mw'] for ts in target_valid],
        val_future_covariates=future_covariates,
        verbose=False
    )

    logger.info(f"{model_collection}:Model fitting completed")

    # Test
    hfc_params = {
        'forecast_horizon': params['h'],
        'stride': int(params['h']/2),
        'retrain': False,
        'overlap_end': False,
        'last_points_only': False,
        'num_samples': params['num_samples']
    }

    his_fcasts = model.historical_forecasts(
        series=[ts['scaled_mw'] for ts in target_test],
        future_covariates=future_covariates,
        **hfc_params
    )

    # Postprocessing
    processed_his_fcasts = []
    for fcast, ts_test in zip(his_fcasts, target_test):
        processed_fcast = []
        for ts in fcast:
            processed_ts = cf.forecast_postprocessor(ts, ts_test, is_train=True, window=params['window'])
            if len(processed_ts) > 0:
                processed_fcast.append(processed_ts)
        if len(processed_fcast) > 0:
            processed_his_fcasts.append(processed_fcast)

    logger.info(f"{model_collection}:Historical forecasts completed")

    # Plot
    metric_series = []
    for station, ts_test, fcast in zip(station_list, target_test, processed_his_fcasts):
        plt.figure(figsize=(28/7*8, 4))
        ts_test['mw'][-28*24:].plot()

        for i, hfc in enumerate(fcast[int(-28*24/params['h']*2):]):
            hfc.plot(label=f"forecast {i}", lw=1)

        fig_name = f"/tmp/historical_forecasts {station} {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.png"
        plt.savefig(fig_name)
        mlflow.log_artifact(local_path=fig_name, artifact_path='Plots')
        plt.close()

        # Test metrics
        list_metrics = []
        for ts in fcast:
            dict_seg_metrics = cf.metrics_segmenter(ts_true=ts_test['mw'], ts_fcst=ts, custom_tag=f"Test {station}", segments=[48, 144])
            list_metrics.append(dict_seg_metrics)

        metric_ser = pd.DataFrame(list_metrics).mean()

        metric_series.append(metric_ser)
        mlflow.log_metrics(metric_ser)

    mlflow.log_metrics(pd.concat(metric_series).rename(index=lambda x: ' '.join(x.split(' ')[2:])).groupby(level=0).sum())
    logger.info(f"{model_collection}:Test metrics logging completed")

    # Log model
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path = 'Model',
        registered_model_name = model_collection,
        signature = infer_signature(
            target_train[0]['mw'].to_dataframe(),
            processed_his_fcasts[0][0].to_dataframe(suppress_warnings=True)
        )
    )
    logger.info(f"{model_collection}:Model logging completed")


def train(
        train_data: dict,
        model: callable,
        params: dict,
        hyperparams: dict,
        model_collection: str
) -> None:
    """Wraps training logic within an MLflow experiment run.

     Encapsulate model training and tag its version in MLflow.

     Args:
        train_data: Training data from `get_data_for_train`, dictionary format.
        model: Model function to be trained.
        params: Model structure parameters, dictionary format.
        hyperparams: Model hyperparameters, dictionary format.
        model_collection: Model name, used for model identification, registration and query.

    Returns:
        None

    Raises:
        Exception: If training or model registration fails.
    """
    # Initialize mlflow client
    client = MlflowClient()

    try:
        with mlflow.start_run(
                run_name=f"{model_collection} {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                tags={
                    'Smooth-Window': str(params['window']),
                    'Lags-Type': params['lags_type'].capitalize(),
                    'Model-Type': params['model_type'].capitalize()
                }
        ):
            model(
                train_data=train_data,
                params=params,
                hyperparams=hyperparams,
                model_collection=model_collection
            )

        # Update model registry
        latest_model = client.get_registered_model(model_collection).latest_versions[0]
        client.set_model_version_tag(
            name=latest_model.name,
            version=latest_model.version,
            key='stage',
            value='Staging'
        )
        logger.info(f"{model_collection}:Model registry update completed")

    except Exception as e:
        logger.error(
            msg=f"{e} with parameters\n"
                f"{[params['lags'], params['tesla_lags'], params['h'], params['window'], params['lags_type'], params['model_type']]} "
                f"{[hyperparams['n_estimators'], hyperparams['learning_rate']]} "
                f"{[hyperparams['max_depth'], hyperparams['reg_alpha'], hyperparams['reg_lambda']]}",
            exc_info=True
        )


def parameter_search(
        ingestor: WindModelV4Data,
        model: callable,
        parameters: dict,
        hyperparameters: dict,
        model_collection: str
) -> None:
    """Perform a grid search over parameter and hyperparameter space.

    Iterates over all parameter combinations to train multiple model candidates.
    Each run is logged and versioned in MLflow.

    Args:
        ingestor: Instance of the `WindModelV4Data` class.
        model: Model function to be trained.
        parameters: Parameter candidate list, dictionary format.
        hyperparameters: Hyperparameter candidate list, dictionary format.
        model_collection: Model name, used for model identification, registration and query.

    Returns:
        None
    """

    # predictwind lags is locked in at this point since we've created the dataset
    predictwind_lags = parameters['predictwind_lags']
    # Traverse parameter space
    for n_estimators, learning_rate in zip(
            hyperparameters['n_estimators'],
            hyperparameters['learning_rate']
    ):
        for (lags, tesla_lags, h, window, lags_type, num_samples, model_type,
             max_depth, reg_alpha, reg_lambda, likelihood, quantiles) in product(
            parameters['lags'],
            parameters['tesla_lags'],
            parameters['h'],
            parameters['window'],
            parameters['lags_type'],
            parameters['num_samples'],
            parameters['model_type'],
            hyperparameters['max_depth'],
            hyperparameters['reg_alpha'],
            hyperparameters['reg_lambda'],
            hyperparameters.get('likelihood', [None]),
            hyperparameters.get('quantiles', [None])
        ):
            # Get unique combination
            params = {
                'lags': lags,
                'tesla_lags': tesla_lags,
                'predictwind_lags': predictwind_lags,
                'h': h,
                'window': window,
                'lags_type': lags_type,
                'num_samples': num_samples,
                'model_type': model_type,
            }

            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'likelihood': likelihood,
                'quantiles': quantiles,
                'early_stopping_rounds': 50,
                'device': 'cuda'
            }

            # Get data
            train_data = get_data_for_train(ingestor=ingestor, params=params, model_collection=model_collection)

            # Train model
            train(
                train_data=train_data,
                model=model,
                params=params,
                hyperparams=hyperparams,
                model_collection=model_collection
            )


def get_new_models(engine: Engine, model_collection: str, criteria: str) -> pd.DataFrame:
    """Get newly trained and production model data from the MLflow database.

    Identify new model candidates (tag is `Staging`) and
    existing production model (tag is `Production`) for relegation and promotion.

    Args:
        engine: SQLAlchemy engine to connect to `MLflow` database.
        model_collection: Model name, used for model identification, registration and query.
        criteria: Criteria for model evaluation (e.g., MAE 48-144).

    Returns:
        A DataFrame with model metadata.

    Raises:
        ValueError: If no staging models are retrieved in the registry.
    """
    logger.info(f"{model_collection}:Getting the newly trained models")

    sql = f"""
    select mv.name, mv.version, mvt.key as tag, mvt.value as stage, m.key as criteria, m.value as score
    from model_versions mv
    join model_version_tags mvt
    on mv.name = mvt.name and mv.version = mvt.version
    join metrics m
    on mv.run_id = m.run_uuid
    where mvt.name = '{model_collection}'
    and mvt.key = 'stage'
    and mvt.value in ('Staging', 'Production')
    and m.key = '{criteria}'
    """
    df_models = pd.read_sql(sql, engine)

    # Check the number of newly trained models
    n_stag = (df_models['stage'] == 'Staging').sum()

    if n_stag:
        logger.info(f"{model_collection}:Retrieved {n_stag} row(s) of newly trained model data")
    else:
        message = f"{model_collection}:No newly trained model data retrieved"
        logger.error(msg=message, exc_info=True)
        raise ValueError(message)

    # Check the number of production models
    n_prod = (df_models['stage'] == 'Production').sum()

    if n_prod > 1:
        logger.warning(f"{model_collection}:Retrieved {n_prod} rows of production model data")

    else:
        logger.info(f"{model_collection}:Retrieved {n_prod} row of production model data")

    return df_models


def relegate_and_promote(df_models: pd.DataFrame) -> None:
    """Promote the best model to Production and updates tags of other models.

    Evaluate and compare models based on error scores.

    If the best model is still `Production`,
    keep it and change the tags of rest models from `Staging` to `Rejected`.

    If the best model is a newly trained model,
    change the tag of the current production model from `Production` to `Deprecated`,
    change the tag of the best model from `Staging` to `Production`,
    change the tags of rest models from `Staging` to `Rejected`.

    Args:
        df_models: A DataFrame of model information returned from `get_new_models`.

    Returns:
        None
    """
    # Initialize mlflow client
    client = MlflowClient()

    # Get the best error
    min_error = df_models['score'].min()

    # Get the latest version with the best error
    latest_version = df_models[df_models['score'] == min_error]['version'].max()

    # Update model tag
    for model in df_models.itertuples(index=False):
        if (model.score != min_error) or (model.score == min_error and model.version != latest_version):

            # Reject newly trained model
            if model.stage == 'Staging':
                client.set_model_version_tag(
                    name=model.name,
                    version=str(model.version),
                    key=model.tag,
                    value='Rejected'
                )
                logger.info(f"{model.name} version {model.version} has been rejected")

            # Deprecate current production model
            else:
                client.set_model_version_tag(
                    name=model.name,
                    version=str(model.version),
                    key=model.tag,
                    value='Deprecated'
                )
                logger.info(f"{model.name} version {model.version} has been deprecated")

        else:
            # Promote newly trained model
            if model.stage == 'Staging':
                client.set_model_version_tag(
                    name=model.name,
                    version=str(model.version),
                    key=model.tag,
                    value='Production'
                )
                logger.info(f"{model.name} version {model.version} has been promoted as production model")

            # Retain current production model
            else:
                logger.info(f"{model.name} version {model.version} has been retained as production model")


def run_experiment(
        ingestor: WindModelV4Data,
        model_collection: str,
        is_prob_model: bool,
        predictwind_lags: list
):
    """Main program entry, running the full model training and selection pipeline.

    Integrating all training processes:
        - Construct parameter combination
        - Run `parameter_search`
        - Get model information
        - Relegate and promote model

    Args:
        ingestor: Instance of the `WindModelV4Data` class.
        model_collection: Model name, used for model identification, registration and query.
        is_prob_model: Whether the model is probabilistic.
        predictwind_lags: Lags used for predictwind future covariates.

    Returns:
        None

    Raises:
        RuntimeError: If any step fails.
    """
    try:
        # Create parameter candidate lists
        parameters = {
            'lags': [
                list(range(-2, -73, -1))
            ],
            'tesla_lags': [
                [0.5] + list(range(3, 12, 3)) + list(range(12, 72, 6)) + list(range(72, 169, 12))
            ],
            'predictwind_lags': predictwind_lags,
            'h': [168],
            'window': [2],
            'lags_type': [
                # 'Point'
                'Sampling',
                # 'non-overlap'
            ],
            'num_samples': [1],
            'model_type': ['deterministic']
        }

        hyperparameters = {
            'n_estimators': [
                100,
                # 300
            ],
            'max_depth': [6],
            'learning_rate': [
                0.3,
                # 0.1
            ],
            'reg_alpha': [1],
            'reg_lambda': [2]
        }

        # Update probabilistic model hyperparameters
        if is_prob_model:
            logger.info(f"{model_collection} model will be trained")

            parameters['window'] = [None]
            parameters['num_samples'] = [100]
            parameters['model_type'] = ['probabilistic']

            hyperparameters['likelihood'] = ['quantile']

            hyperparameters['quantiles'] = [[0.05, 0.25, 0.5, 0.75, 0.95]]
            criteria = 'MCRPS 48-144'

        else:
            logger.info(f"{model_collection} model will be trained")

            criteria = 'MAE 48-144'

        # Train
        parameter_search(
            ingestor=ingestor,
            model=global_xgboost,
            parameters=parameters,
            hyperparameters=hyperparameters,
            model_collection=model_collection
        )

        # Get newly trained models
        df_models = get_new_models(
            engine=mlflow_eng,
            criteria=criteria,
            model_collection=model_collection
        )

        # Select the best model for production
        relegate_and_promote(df_models)

    except Exception as e:
        logger.error(e, exc_info=True)
        raise RuntimeError(e)




if __name__ == '__main__':

    # Run the following command in terminal instead of python

    # Dev training
    #  mlflow server --default-artifact-root s3://mlflow-data-3240/ --backend-store-uri postgresql://emhapp@trading-dev.cduqymuka03q.ap-southeast-2.rds.amazonaws.com:5432/mlflow_dev
    # Prod training
    # AWS_PROFILE=wennan-mkts mlflow server --default-artifact-root s3://mlflow-data-8916/ --backend-store-uri postgresql://wennan@trading-prod.cuvldknbzcxv.ap-southeast-2.rds.amazonaws.com:5432/mlflow

    is_prob_model = True

    if is_prob_model:
        model_collection = 'Probabilistic_XGBoost'
    else:
        model_collection = 'Deterministic_XGBoost'

    # Initialize mlflow
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(f"{model_collection}_Experiment")

    # Initialize class, set to training mode
    ingestor = WindModelV4Data(
        is_train=True,
        lag_buffer_days=2,
        engine=gauss_eng
    )

    predictwind_lags = [.5, 12, 48, 120]

    # Get data from database
    ingestor.ingest_from_database(
        forecast_model=model_collection,
        train_start_date='2022-05-01',
        fcast_timestamp=None,
        predictwind_lags=predictwind_lags,
        dump_file_path='/tmp'
    )

    # Get data from files
    # ingestor.ingest_from_files(
    #     dump_file_path='/tmp',
    #     predictwind_lags=predictwind_lags
    # )

    run_experiment(
        ingestor=ingestor,
        model_collection=model_collection,
        is_prob_model=is_prob_model,
        predictwind_lags=predictwind_lags
    )

