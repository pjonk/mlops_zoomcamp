import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")
mlflow.autolog()


def extract_data(year: int, month: int) -> tuple[pd.DataFrame]:
    """ Download training and validation data, where the next month's data is used for validation. """
    
    def download_data(year: int, month: int) -> pd.DataFrame:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
        return pd.read_parquet(url)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_train = download_data(year, month)
    df_val = download_data(next_year, next_month)

    return df_train, df_val
    

def transform_data(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[np.array]:
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
    
    print(f'Total rows in raw training data: {len(df_train.index)}')

    df_train = filter_df(df_train)
    df_val = filter_df(df_val)

    print(f'Total rows in transformed training data: {len(df_train.index)}')

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, y_train, X_val, y_val


def train_model(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> str:
    """ Fit a linear regression model. """

    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        print(f'intercept: {lr.intercept_}')

        train_rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric('train_rmse', train_rmse)

        y_pred = lr.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('val_rmse', val_rmse)

    return run.info.run_id


def load_model(run_id: str):
    """ Store model in the mlflow registry. """

    mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="linear_regression_model")


def run_pipeline(year: int, month: int):
    
    df_train, df_val = extract_data(year, month)
    X_train, y_train, X_val, y_val = transform_data(df_train, df_val)
    run_id = train_model(X_train, y_train, X_val, y_val)
    load_model(run_id)


if __name__ == "__main__":

    run_pipeline(2023, 3)





