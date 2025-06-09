#!/usr/bin/env python
# coding: utf-8

# Week 3 script: Ride duration prediction


import pandas as pd
import pickle
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb

import mlflow

from prefect import flow, task


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(year, month, dataset_type):
    """
    Read data from a URL.
    The data is taken from the page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
    Working with "Taxi Trip Records", Yellow or Green, depending on the argument.

    Args:
        year (int): The year of the data.
        month (int): The month of the data.

    Returns:
        pd.DataFrame: The dataframe with the data.
    """
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset_type}_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    print(
        f"Read dataset from {url}, i.e. {dataset_type} taxi trip data for {year}-{month:02d}. Number of rows: {len(df)}"
    )

    if dataset_type == "green":
        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    else:
        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    print(f"Number of rows after preprocessing: {len(df)}")

    return df


@task
def create_X(
    df: pd.DataFrame,
    dv: sklearn.feature_extraction.DictVectorizer = None,
    categorical: list[str] = None,
    use_numerical: bool = True,
) -> tuple[scipy.sparse._csr.csr_matrix, sklearn.feature_extraction.DictVectorizer]:
    """
    Create a feature matrix from a dataframe.
    """
    numerical = ["trip_distance"] if use_numerical else []
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    X_val: scipy.sparse._csr.csr_matrix,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> str:
    """
    Train a model with best hyperparams and write everything out
    """
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@task(log_prints=True)
def train_linear_regression(
    X_train: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    X_val: scipy.sparse._csr.csr_matrix,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> str:
    """
    Train a linear regression model and write everything out
    """
    with mlflow.start_run() as run:

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_params(model.get_params())

        # Log intercept and coefficients of the model
        mlflow.log_param("intercept", model.intercept_)
        mlflow.log_param("coefficients", model.coef_.tolist())

        Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Register the model in MLflow
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id


@flow
def run(year: int, month: int, dataset_type: str) -> str:
    """
    Run the training pipeline.
    """
    df_train = read_dataframe(year=year, month=month, dataset_type=dataset_type)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_val = read_dataframe(year=next_year, month=next_month, dataset_type=dataset_type)

    if dataset_type == "green":
        categorical = ["PU_DO"]
        use_numerical = True
    else:
        categorical = ["PULocationID", "DOLocationID"]
        use_numerical = False

    X_train, dv = create_X(
        df_train, categorical=categorical, use_numerical=use_numerical
    )
    X_val, _ = create_X(
        df_val, dv, categorical=categorical, use_numerical=use_numerical
    )

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    if dataset_type == "green":
        run_id = train_best_model(X_train, y_train, X_val, y_val, dv)
        print(f"MLflow run_id: {run_id}")
    else:
        run_id = train_linear_regression(X_train, y_train, X_val, y_val, dv)
        print(f"MLflow run_id: {run_id}")

    return run_id


@flow
def main_flow() -> None:
    """
    Main function to run the training pipeline.
    """
    year = 2023
    month = 1
    dataset_type = "green"
    run_id = run(year=year, month=month, dataset_type=dataset_type)

    with open("run_id.txt", "w") as f:
        f.write(run_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model to predict taxi trip duration."
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year of the data to train on"
    )
    parser.add_argument(
        "--month", type=int, required=True, help="Month of the data to train on"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Type of the dataset to train on (yellow or green)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    run_id = run(year=args.year, month=args.month, dataset_type=args.dataset_type)

    with open("run_id.txt", "w") as f:
        f.write(run_id)
