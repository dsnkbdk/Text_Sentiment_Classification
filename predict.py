import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

import io
import json
import mlflow
import logging
import numpy as np
import pandas as pd
import custom_functions as cf

from ast import literal_eval
from darts import TimeSeries
from ingest import WindModelV4Data
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from common import get_database_url, sns_publish_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('predict')
logger.setLevel(logging.INFO)

SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
GAUSS_DATABASE_SECRET_ID = os.getenv('GAUSS_DATABASE_SECRET_ID')
MLFLOW_DATABASE_SECRET_ID = os.getenv('MLFLOW_DATABASE_SECRET_ID')


def get_mlflow_models(engine: Engine, model_collection: str, stage: str = 'Production') -> pd.DataFrame:
    """Get the current production model from the MLflow registry.

    Args:
        engine: SQLAlchemy engine to connect to `MLflow` database.
        model_collection: Model name, used for model identification, registration and query.
        stage: Different stages of the model, default is `Production`.
    Returns:
        A DataFrame containing model metadata.
    Raises:
        ValueError: If no matching models are retrieved.
    """
    logger.info(f"{model_collection}:Getting current production model(s)")

    sql = f"""
    select mv.name, mv.version, mv.source, mv.run_id
    from model_versions mv
    join model_version_tags mvt
    on mv.name = mvt.name and mv.version = mvt.version
    where mvt.name = '{model_collection}'
    and mvt.key = 'stage'
    and mvt.value = '{stage}'
    """
    df_models = pd.read_sql(sql, engine)

    if not df_models.empty:
        logger.info(f"{model_collection}:Retrieved {len(df_models)} row(s) of production model data")
    else:
        message = f"{model_collection}:No production model data retrieved"
        logger.error(msg=message, exc_info=True)
        raise ValueError(message)

    return df_models


def get_model_params(engine: Engine, model_collection: str, run_id: str) -> dict:
    """Get the parameters of the current production model.

    Args:
        engine: SQLAlchemy engine to connect to `MLflow` database.
        model_collection: Model name, used for model identification, registration and query.
        run_id: Retrieve the parameters based on the model id obtained from `get_mlflow_models`.

    Returns:
        A dictionary containing production model parameters.

    Raises:
        ValueError: If no parameter data is retrieved.
    """
    logger.info(f"{model_collection}:Getting model parameters")

    sql = f"""
    select key, value
    from params
    where run_uuid = '{run_id}'
    """
    df_params = pd.read_sql(sql, engine).set_index('key')

    if not df_params.empty:
        logger.info(f"{model_collection}:Retrieved {len(df_params)} row(s) of parameter data")
    else:
        message = f"{model_collection}:No parameter data retrieved"
        logger.error(msg=message, exc_info=True)
        raise ValueError(message)

    dict_params = df_params['value'].to_dict()

    return dict_params


def get_data_for_predict(
        engine: Engine,
        model_collection: str,
        fcast_timestamp: str,
        dict_params: dict
) -> dict[str, dict[str, dict[str, TimeSeries]]]:
    """Get structured data for forecasting.

    Initialize and call the `WindModelV4Data` class to get the data for forecasting.

    Args:
        engine: SQLAlchemy engine to connect to `Gauss` database.
        model_collection: Model name, used for model identification, registration and query.
        fcast_timestamp: Customize the forecast start time T0.
        dict_params: Model parameters obtained from `get_model_params`.

    Returns:
        A nested dictionary of TimeSeries objects for forecasting.
    """
    logger.info(f"{model_collection}:Getting data for prediction")

    # Initialize class, set to testing mode
    ingestor = WindModelV4Data(
        is_train=False,
        lag_buffer_days=2,
        engine=engine
    )

    ingestor.ingest_from_database(
        forecast_model=model_collection,
        fcast_timestamp=fcast_timestamp,
        predictwind_lags=literal_eval(dict_params['predictwind_lags'])
    )

    # Get data for predict
    predict_data = ingestor.data_integration_output(
        tesla_lags=literal_eval(dict_params['tesla_lags']),
        h=literal_eval(dict_params['h'])
    )

    return predict_data


def predict(
        model_collection: str,
        dict_model: dict,
        dict_params: dict,
        predict_data: dict,
        fcast_timestamp: str = None
) -> tuple[list[str], list[TimeSeries]]:
    """Execute the forecast process.

    Loads the production model from MLflow, execute forecast, postprocess, and check the forecast periods.
    Aggregates NZL forecasts for probabilistic models.

    Args:
        model_collection: Model name, used for model identification, registration and query.
        dict_model: Model information in dictionary format from `df_models`.
        dict_params: Model parameters obtained from `get_model_params`.
        predict_data: Data for forecasting, from `get_data_for_predict`, dictionary format.
        fcast_timestamp: Customize the forecast start time T0.

    Returns:
        Tuple of station codes and their forecast TimeSeries.
    """
    logger.info(f"{model_collection}:Loading {dict_model['name']} version {dict_model['version']}")

    # Get model
    model = mlflow.sklearn.load_model(dict_model['source'])

    # Unpack data
    future_covariates = [station_data['future_covariates'] for station_data in predict_data['data'].values()]
    target_fcast = [station_data['target_fcast'] for station_data in predict_data['data'].values()]
    station_list = list(predict_data['data'].keys())

    # Forecast
    logger.info(f"{model_collection}:Start forecasting")

    pred = []
    for station, ts_target, ts_futu in zip(station_list, target_fcast, future_covariates):
        try:
            logger.info(f"{model_collection}:The actual mw for station {station} start at '{ts_target.start_time().tz_localize('UTC')}' and end at '{ts_target.end_time().tz_localize('UTC')}'")

            if model_collection == 'Deterministic_XGBoost':
                num_samples = literal_eval(dict_params['num_samples'])
            else:
                num_samples = literal_eval(dict_params['num_samples']) * 10

            station_pred = model.predict(
                series=ts_target['scaled_mw'],
                future_covariates=ts_futu,
                n=literal_eval(dict_params['h']),
                num_samples=num_samples
            )

            # Postprocessing
            station_pred = cf.forecast_postprocessor(
                ts_pred=station_pred,
                ts_target=ts_target,
                is_train=False,
                window=literal_eval(dict_params['window'])
            )

            pred.append(station_pred)

            # Check forecast time period
            fcast_start_time = station_pred.start_time().tz_localize('UTC')
            fcast_end_time = station_pred.end_time().tz_localize('UTC')

            logger.info(f"{model_collection}:The forecasts for station {station} start at '{fcast_start_time}' and end at '{fcast_end_time}'")

            if fcast_timestamp is not None:
                fcast_timestamp_utc = pd.Timestamp(fcast_timestamp, tz='UTC')

                if fcast_start_time > fcast_timestamp_utc or fcast_end_time < fcast_timestamp_utc + pd.Timedelta(hours=literal_eval(dict_params['h'])-1):
                    message = f"{model_collection}:The forecasts for station {station} does not span the full forecast period"

                    logger.warning(message)
                    sns_publish_error(
                        topic_arn=SNS_TOPIC_ARN,
                        subject='wind_model_v4.predict.py',
                        message=f"Warning\n\n{message}"
                    )

        except Exception as e:
            message = f"{e}:The forecast for station {station} failed"

            logger.warning(message)
            sns_publish_error(
                topic_arn=SNS_TOPIC_ARN,
                subject='wind_model_v4.predict.py',
                message=f"Warning\n\n{message}"
            )
            continue

    # Aggregate NZ total samples
    if dict_params['model_type'] == 'probabilistic':
        min_start = min(p.start_time() for p in pred)
        max_end = max(p.end_time() for p in pred)

        nzl_pred = []

        for p in pred:
            f = p.freq.delta.total_seconds()

            if p.start_time() > min_start:
                n_steps = int((p.start_time() - min_start).total_seconds() / f)
                p = p.prepend_values(np.zeros((n_steps, p.n_components, p.n_samples)))

            if p.end_time() < max_end:
                n_steps = int((max_end - p.end_time()).total_seconds() / f)
                p = p.append_values(np.zeros((n_steps, p.n_components, p.n_samples)))

            nzl_pred.append(p)

        station_list.append('NZL')
        pred.append(sum(nzl_pred))

    logger.info(f"{model_collection}:Forecast completed")

    return station_list, pred

# TODO - refactor. We only use the table emh_forecast_wind with cols:
# model_collection, series_name, trading_date, trading_period, period_timestamp, forecast_mw, prediction_insert_datetime
# we don't need to repeat the model collection in the series name.
# possibly we could refactor emh_wind_model_prediction to just have model_collection, model_version, forecast_datetime
def format(
        model_collection: str,
        dict_model: dict,
        dict_params: dict,
        station_list: list,
        pred: list,
        quantiles: list = None,
        fcast_timestamp: str = None
) -> pd.DataFrame:
    """Format forecast results into a structured DataFrame for downstream insertion.

    Convert forecasts to DataFrame, apply time zone and trading period,
    melt quantile forecasts for probabilistic models, and filters results based on T0.

    Args:
        model_collection: Model name, used for model identification, registration and query.
        dict_model: Model information in dictionary format from `df_models`.
        dict_params: Model parameters obtained from `get_model_params`.
        station_list: List of station from `predict`.
        pred: List of forecast from `predict`.
        quantiles: Convert samples to quantiles df.
        fcast_timestamp: Customize the forecast start time T0.

    Returns:
         A formatted forecast DataFrame ready for database insertion.
    """
    logger.info(f"{model_collection}:Start formatting forecast data")

    # Convert to df
    dfs = []
    for (station, ts) in zip(station_list, pred):
        # For probabilistic model, convert to quantiles df
        if dict_params['model_type'] == 'probabilistic':

            if quantiles is None:
                quantiles = literal_eval(dict_params['quantiles'])

            df = ts.quantiles_df(quantiles).reset_index()

        # For deterministic model, convert to regular df
        else:
            df = ts.to_dataframe().reset_index()

        df['model_name'] = f"{station}"
        df.columns.name = None

        dfs.append(df)

    df_pred = (
        pd.concat(dfs, ignore_index=True)
        .assign(
            model_version = dict_model['version'],
            period_timestamp = lambda df: pd.to_datetime(df['period_timestamp'], utc=True).dt.tz_convert('Pacific/Auckland'),
            trading_date = lambda df: df['period_timestamp'].dt.date,
            trading_period = lambda df: (df['period_timestamp'].dt.hour * 60 + df['period_timestamp'].dt.minute) // 30 + 1,
            insert_datetime = pd.Timestamp.now(tz='Pacific/Auckland'),
        )
    )

    # Fix trading period for NZDT and NZST conversion
    df_pred = df_pred.groupby(by=['model_name', 'trading_date'], group_keys=False).apply(
        lambda group: cf.fix_trading_period(
            group=group,
            freq=60
        )
    )

    if fcast_timestamp is not None:
        fcast_timestamp_utc = pd.Timestamp(fcast_timestamp, tz='UTC')

        # Only keep forecasts equal to or later than T0
        df_pred = df_pred[df_pred['period_timestamp'].dt.tz_convert('UTC') >= fcast_timestamp_utc]

    # Melt quantile columns
    if dict_params['model_type'] == 'probabilistic':
        quantile_cols = [col for col in df_pred.columns if col.startswith('forecast_mw_')]

        df_pred = df_pred.melt(
            id_vars=['period_timestamp', 'model_name', 'model_version', 'trading_date', 'trading_period', 'insert_datetime'],
            value_vars=quantile_cols,
            var_name='quantile',
            value_name='forecast_mw'
        )

        df_pred['suffix'] = (df_pred['quantile'].str.split('_').str[-1].astype(float) * 100).astype(int)
        df_pred['suffix'] = df_pred['suffix'].apply(lambda x: f"{x:02d}")  # Format with leading zeros
        df_pred['model_name'] = df_pred['model_name'] + '_p' + df_pred['suffix']

    df_pred['model_collection'] = model_collection

    logger.info(f"{model_collection}:Forecast data formatting completed")

    return df_pred

# TODO refactor this. It should only insert, not do some of the aggregation and formatting.
def insert_series_aggregations(model_collection: str, dict_params: dict, engine: Engine, df_pred: pd.DataFrame) -> None:
    """Insert formatted forecast data into the database.

    Performs additional checks for missing or duplicate stations,
    aggregates NZL forecasts, and insert the results into the table `emh_forecast_wind`.

    Args:
        model_collection: Model name, used for model identification, registration and query.
        dict_params: Model parameters obtained from `get_model_params`.
        engine: Engine to connect to `Gauss` database.
        df_pred: Formatted forecast data obtained from `predict()`.

    Raises:
        ValueError: If duplicate stations are found in deterministic forecasts.
    """
    logger.info(f"{model_collection}:Start inserting aggregate forecast data into table emh_forecast_wind")

    df_pred = df_pred.copy()
    df_pred = (
        df_pred
        .rename(columns={'model_name': 'series_name', 'insert_datetime': 'prediction_insert_datetime'})
        .assign(
            station = lambda df: df['series_name'].str.split('_').str[0]
        )
    )

    # col order for db
    col_order = [
        'model_collection',
        'series_name',
        'trading_date',
        'trading_period',
        'period_timestamp',
        'forecast_mw',
        'prediction_insert_datetime'
    ]


    if dict_params['model_type'] == 'probabilistic':
        df = df_pred[col_order]

    else:
        stations = df_pred['station'].unique()
        tp_groups = df_pred.groupby(['model_collection', 'model_version', 'trading_date', 'trading_period', 'period_timestamp'])

        # TODO put this somewhere more appropriate. Not at insert time. Also why are we only doing it for deterministic
        # check for dupes or missing stations
        for _, group in tp_groups:
            station_series = group['station']
            duplicate_station = station_series[station_series.duplicated()].unique()

            if duplicate_station.size > 0:
                message = f"{model_collection}: Duplicate wind farm in forecasts: {duplicate_station}"
                logger.error(msg=message, exc_info=True)
                raise ValueError(message)

            missing_station = set(stations) - set(station_series)
            if missing_station:
                logger.info(f"{model_collection}:Missing wind farm in forecast: {sorted(missing_station)}")

        # TODO put this with the probabalistic aggregation in predict() or postprocess
        # aggregate NZL series
        df_NZL = (
            tp_groups['forecast_mw'].sum().reset_index()
            .assign(
                series_name = lambda df: 'NZL',
                prediction_insert_datetime = pd.Timestamp.now(tz='Pacific/Auckland'),
            )
        )

        df = pd.concat([df_pred[col_order], df_NZL[col_order]], ignore_index=True)

    # Insert into emh_forecast_wind
    buf = io.StringIO(df.to_csv(index=False))

    with engine.begin() as eng:
        dbapi = eng.connection
        curs = dbapi.cursor()
        curs.copy_expert('copy emh_forecast_wind from stdin with csv header', file=buf)

    logger.info(f"{model_collection}:Successfully inserted {curs.rowcount} rows into table emh_forecast_wind")


def lambda_handler(event, context = None):
    """Main program entry, integrating all forecast processes, logging and publishing exceptions.

    Handles model retrieval, parameter loading, forecast execution, result formatting, and database insertion.

    Args:
        event: JSON file used to start `lambda_handler`, contains `model_collection` and `fcast_timestamp`.
        context: AWS Lambda context object (optional).

    Raises:
        RuntimeError: If any step fails.
    """
    model_collection = event['model_collection']
    logger.info(f"Start forecasting with model {model_collection}")

    mlflow_url = get_database_url(MLFLOW_DATABASE_SECRET_ID)
    gauss_url = get_database_url(GAUSS_DATABASE_SECRET_ID)
    mlflow_eng = create_engine(mlflow_url)
    gauss_eng = create_engine(gauss_url)

    try:
        # Get model info
        df_models = get_mlflow_models(
            engine=mlflow_eng,
            model_collection=model_collection,
            stage='Production'
        )

        if len(df_models) > 1:
            latest_version = df_models['version'].max()
            df_models = df_models[df_models['version'] == latest_version]
            message = f"{model_collection}:Multiple production models were retrieved, keep the latest version {latest_version}"

            logger.warning(message)
            sns_publish_error(
                topic_arn=SNS_TOPIC_ARN,
                subject='wind_model_v4.predict.py',
                message=f"Warning\n\n{message}")

        # Convert model info into a dict
        dict_model = df_models.iloc[0].to_dict()

        # Get model params
        dict_params = get_model_params(engine=mlflow_eng, model_collection=model_collection, run_id=dict_model['run_id'])
        logger.info(f"{model_collection}:The model parameters:\n{json.dumps(dict_params, indent=2)}")

        # Define forecast start time T0
        fcast_timestamp = event.get(
            'fcast_timestamp',
            pd.Timestamp.now(tz='Pacific/Auckland').ceil('h').strftime('%Y-%m-%d %H:%M:%S %z')
        )

        # Get data for predict
        predict_data = get_data_for_predict(
            engine=gauss_eng,
            model_collection=model_collection,
            fcast_timestamp=fcast_timestamp,
            dict_params=dict_params
        )

        station_list, pred = predict(
            model_collection=model_collection,
            dict_model=dict_model,
            dict_params=dict_params,
            predict_data=predict_data,
            fcast_timestamp=fcast_timestamp
        )

        df_pred = format(
            model_collection=model_collection,
            dict_model=dict_model,
            dict_params=dict_params,
            station_list=station_list,
            pred=pred,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            fcast_timestamp=fcast_timestamp
        )

        insert_series_aggregations(
            model_collection=model_collection,
            dict_params=dict_params,
            engine=gauss_eng,
            df_pred=df_pred
        )

    except Exception as e:
        logger.error(e, exc_info=True)
        sns_publish_error(
            topic_arn=SNS_TOPIC_ARN,
            subject='wind_model_v4.predict.py',
            message=f"Error\n\n{e}"
        )
        raise RuntimeError(e)

    finally:
        matplot_path = os.environ['MPLCONFIGDIR']

        for path in [matplot_path, '/tmp']:
            if os.path.exists(path):
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))



if __name__ == '__main__':

    event = {
        "model_collection": "Deterministic_XGBoost",
        # "model_collection": "Probabilistic_XGBoost",
        # "fcast_timestamp": "2025-03-01 22:00:00 +1300"
    }
    lambda_handler(event=event)

