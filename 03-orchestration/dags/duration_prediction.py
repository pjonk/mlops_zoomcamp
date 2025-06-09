from airflow.decorators import dag, task
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import psutil


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['week_3'],
    description='Training pipeline for week 3 of the MLOPS-zoomcamp.',
)
def duration_prediction_training_pipeline():

    @task
    def extract_data(year: int, month: int) -> dict:
        """ Download training and validation data, where the next month's data is used for validation. """
        
        def download_data(year: int, month: int) -> pd.DataFrame:
            url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
            return pd.read_parquet(url)

        next_year = year if month < 12 else year + 1
        next_month = month + 1 if month < 12 else 1

        df_train = download_data(year, month)
        df_val = download_data(next_year, next_month)

        train_path = "/tmp/train.parquet"
        val_path = "/tmp/val.parquet"
        
        df_train.to_parquet(train_path)
        df_val.to_parquet(val_path)

        return {"train_path": train_path, "val_path": val_path}

        
    @task
    def transform_data(train_path: str, val_path: str) -> dict:
        """ Add duration, remove outliers, select features, vectorize. """

        def filter_df(df: pd.DataFrame) -> pd.DataFrame:
            df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
            df.duration = df.duration.dt.total_seconds() / 60

            df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

            return df

        def create_X(df: pd.DataFrame, dv: DictVectorizer=None) -> tuple[pd.DataFrame, DictVectorizer]:
            features = ['PULocationID', 'DOLocationID']
            df[features] = df[features].astype(str)
            train_dicts = df[features].to_dict(orient='records')

            if dv is None:
                dv = DictVectorizer(sparse=True)
                X = dv.fit_transform(train_dicts)
            else:
                X = dv.transform(train_dicts)

            return X, dv
        
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)

        print(f'Total rows in raw training data: {len(df_train.index)}')

        df_train = filter_df(df_train)
        df_val = filter_df(df_val)

        print(f'Total rows in transformed training data: {len(df_train.index)}')

        X_train, dv = create_X(df_train)
        X_val, _ = create_X(df_val, dv)

        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values
    
        # Prepare output directory
        output_dir = Path('/opt/airflow/dags/temp')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs to files to avoid large XComs
        X_train_path = output_dir / 'X_train.joblib'
        y_train_path = output_dir / 'y_train.joblib'
        X_val_path = output_dir / 'X_val.joblib'
        y_val_path = output_dir / 'y_val.joblib'
        dv_path = output_dir / 'dv.joblib'

        joblib.dump(X_train, X_train_path)
        joblib.dump(y_train, y_train_path)
        joblib.dump(X_val, X_val_path)
        joblib.dump(y_val, y_val_path)
        joblib.dump(dv, dv_path)

        return {
            'X_train_path': str(X_train_path),
            'y_train_path': str(y_train_path),
            'X_val_path': str(X_val_path),
            'y_val_path': str(y_val_path),
            'dv_path': str(dv_path)
        }

    @task
    def train_model(paths: dict) -> str:
        """ Fit a linear regression model. """

        import mlflow

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("week-3-duration-prediction")

        # Load data from files
        X_train = joblib.load(paths['X_train_path'])
        y_train = joblib.load(paths['y_train_path'])
        X_val = joblib.load(paths['X_val_path'])
        y_val = joblib.load(paths['y_val_path'])

        print(f"Available memory: {psutil.virtual_memory().available / (1024**2):.2f} MB")

        with mlflow.start_run() as run:
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            mlflow.log_params(lr.get_params())
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.sklearn.log_model(lr, artifact_path="model")

            y_pred = lr.predict(X_train)
            print(f'intercept: {lr.intercept_}')

            train_rmse = root_mean_squared_error(y_train, y_pred)
            mlflow.log_metric('train_rmse', train_rmse)

            y_pred = lr.predict(X_val)
            val_rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric('val_rmse', val_rmse)

        return {'run_id': run.info.run_id}

    @task
    def load_model(run_info: dict):
        """ Store model in the mlflow registry. """

        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        mlflow.register_model(model_uri=f"runs:/{run_info['run_id']}/model", name="linear_regression_model")

    
    # DAG
    paths= extract_data(2023, 3)
    prepared_data_paths = transform_data(paths['train_path'], paths['val_path'])
    run_id = train_model(prepared_data_paths)
    load_model(run_id)

duration_prediction_training_pipeline()
