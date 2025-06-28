from datetime import datetime
import pandas as pd
import pandas.testing as pdt
import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_read_data():

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df = pd.DataFrame(data, columns=columns)

    categorical = ["PULocationID", "DOLocationID"]

    df = batch.prepare_data(df=df, categorical=categorical)

    reference_df = pd.DataFrame({
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

    reference_df["tpep_pickup_datetime"] = pd.to_datetime(reference_df["tpep_pickup_datetime"])
    reference_df["tpep_dropoff_datetime"] = pd.to_datetime(reference_df["tpep_dropoff_datetime"])

    pdt.assert_frame_equal(df, reference_df)
