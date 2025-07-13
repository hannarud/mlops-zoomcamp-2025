#!/usr/bin/env python
# coding: utf-8

# Week 4 homework script

import sys
import pickle
import numpy as np
import pandas as pd


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def run():
    taxi_type = sys.argv[1]  # 'yellow'
    year = int(sys.argv[2])  # 2023
    month = int(sys.argv[3])  # 3

    print(f"Processing {taxi_type} data for {year}-{month}")

    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(np.mean(y_pred))

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    output_file = f"{taxi_type}-{year:04d}-{month:02d}.parquet"

    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    run()
