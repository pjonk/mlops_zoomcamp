import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import scipy

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Callable


def load_vectorizer_and_model(model_path: str, vectorizer_path: str) -> tuple[DictVectorizer, LinearRegression]:

    print(f'Loading vectorizer and model from {vectorizer_path}, {model_path}...')

    with open(vectorizer_path, 'rb') as f_in:
        dv = pickle.load(f_in)

    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)

    return dv, model


def read_data(create_url: Callable, year: int, month: int) -> pd.DataFrame:

    print(f'Reading yellow taxi data for year {year:04d} and month {month:02d}...')

    url = create_url(year, month)
    df = pd.read_parquet(url)
    
    return df


def transform_data(df: pd.DataFrame, dv: DictVectorizer) -> scipy.sparse.csr_matrix:

    print('Vectorizing data...')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    return X_val, df


def make_predictions(X_val: scipy.sparse.csr_matrix, model: LinearRegression) -> np.array:

    print('Making predictions...')
    y_pred = model.predict(X_val)

    return y_pred


def store_results(preds: np.array, df: pd.DataFrame, result_folder: str, year: int, month: int):

    output_folder = Path(result_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f'{year:04d}-{month:02d}_predictions.parquet' 

    print(f'Storing predictions with added ride_id to {output_file}...')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = preds

    df_result = df[['ride_id', 'predictions']]

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def main():

    parser = argparse.ArgumentParser(
                        prog='`Duration Prediction',
                        description='Predict the duration of NY taxi rides.',
                    )

    parser.add_argument('-m', '--month')
    parser.add_argument('-y', '--year')
    parser.add_argument('-p', '--model_path', default='./artifacts/model.bin')
    parser.add_argument('-v', '--vectorizer_path', default='./artifacts/dv.bin')
    parser.add_argument('-r', '--result_path', default='./data')
    args = parser.parse_args()

    year = int(args.year)
    month = int(args.month)

    def create_url(year: str, month: str) -> str:
        return f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(create_url, year, month)
    dv, lr_model = load_vectorizer_and_model(args.model_path, args.vectorizer_path)
    X_val, transformed_df = transform_data(df, dv)
    preds = make_predictions(X_val, lr_model)
    print(f'The mean of the predictions is: {preds.mean().round(2)}')
    store_results(preds=preds, df=transformed_df, result_folder=args.result_path, year=year, month=month)


if __name__ == '__main__':
    main()