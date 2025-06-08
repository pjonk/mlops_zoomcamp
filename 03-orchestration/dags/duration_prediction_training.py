from airflow.decorators import dag, task
from datetime import datetime
from pathlib import Path
import pandas as pd
import xgboost as xgb
import mlflow
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@dag(
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nyc-taxi', 'ml', 'xgboost'],
    description='Train XGBoost model on NYC Taxi data using TaskFlow API',
)
def nyc_taxi_train_pipeline():

    @task
    def read_dataframe(year: int, month: int):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        return df

    @task
    def create_X(df, dv=None):
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')

        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)

        return X, dv

    @task
    def train_model(X_train, y_train, X_val, y_val, dv):
        with mlflow.start_run() as run:
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            best_params = {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:linear',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }

            mlflow.log_params(best_params)

            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

            return run.info.run_id

    @task
    def write_run_id(run_id: str):
        with open("run_id.txt", "w") as f:
            f.write(run_id)

    @task
    def get_params():
        # You can fetch from Variable, Param, or provide fixed values
        return {"year": 2023, "month": 1}

    # ----- DAG Structure -----
    params = get_params()
    df_train = read_dataframe(params["year"], params["month"])
    
    next_month = (params["month"] % 12) + 1
    next_year = params["year"] + (params["month"] // 12)

    df_val = read_dataframe(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    target = 'duration'
    y_train = df_train.map(lambda df: df[target].values)
    y_val = df_val.map(lambda df: df[target].values)

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    write_run_id(run_id)


nyc_taxi_train_pipeline()
