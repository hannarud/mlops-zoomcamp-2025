# coding: utf-8

# Week 6 homework tests

from datetime import datetime
import pandas as pd
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df.to_parquet(
        "test_parquet.parquet",
        engine='pyarrow',
        compression=None,
        index=False,
    )

    expected_processed_data = [
        ("-1", "-1", dt(1, 1), dt(1, 10), 9.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0),     
    ]

    expected_processed_df = pd.DataFrame(expected_processed_data, columns=columns + ['duration'])

    df_processed = prepare_data(df)
    
    print(df_processed.to_dict(orient="records"))
    print(expected_processed_df.to_dict(orient="records"))
    print(df_processed.equals(expected_processed_df))

    assert df_processed.shape == expected_processed_df.shape
    assert df_processed.equals(expected_processed_df)
    assert df_processed.to_dict(orient="records") == expected_processed_df.to_dict(orient="records")
