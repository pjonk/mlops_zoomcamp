import pandas as pd
from batch import get_input_path
import os
import sys


def upload_fake_data():

    df_input = pd.DataFrame({
            "PULocationID": {0: "-1", 1: "1"},
            "DOLocationID": {0: "-1", 1: "1"},
            "tpep_pickup_datetime": {
                0: "2023-01-01 01:01:00",
                1: "2023-01-01 01:02:00",
            },
            "tpep_dropoff_datetime": {
                0: "2023-01-01 01:10:00",
                1: "2023-01-01 01:10:00",
            },
            "duration": {0: 9.0, 1: 8.0},
        })


    endpoint_url = os.getenv('S3_ENDPOINT_URL', None)

    if endpoint_url:
        options = {
        'client_kwargs': {
            'endpoint_url': endpoint_url
            }
        }  

        input_path = get_input_path(2023, 1)
        print(f'Saving data to: {input_path}')    

        df_input.to_parquet(
            input_path,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
    else: 
        print('No endpoint set!')


def test_batch_job():

    exit_code = os.system("python batch.py 2023 01")

    return exit_code


if __name__ == "__main__":
    upload_fake_data()
    test_batch_job()
    