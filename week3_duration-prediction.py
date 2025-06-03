#!/usr/bin/env python
# coding: utf-8

# Week 3 script: Ride duration prediction


import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import root_mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

# ## Data
#
# Taken from the page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
#
# Downloading "Green Taxi Trip Records" for January and February, 2023.

# !mkdir data

# Green Taxi Trip Records for January, 2023
# !wget -O data/green_tripdata_2023-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet

# Green Taxi Trip Records for February, 2023
# !wget -O data/green_tripdata_2023-02.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet


def read_dataframe(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


train_data_path = "./data/green_tripdata_2023-01.parquet"
val_data_path = "./data/green_tripdata_2023-02.parquet"

df_train = read_dataframe(train_data_path)
df_val = read_dataframe(val_data_path)

# ## Baseline model

len(df_train), len(df_val)

categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
numerical = ["trip_distance"]

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)

target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)

with open("models/lin_reg.bin", "wb") as f_out:
    pickle.dump((dv, lr), f_out)

with mlflow.start_run():

    mlflow.set_tag("developer", "hanna")

    mlflow.log_param("train-data-path", train_data_path)
    mlflow.log_param("val-data-path", val_data_path)

    alpha = 0.1

    mlflow.log_param("alpha", alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")

# ## XGBoost

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

    return {"loss": rmse, "status": STATUS_OK}


search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "reg:linear",
    "seed": 42,
}

best_result = fmin(
    fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=Trials()
)

mlflow.xgboost.autolog(disable=True)

with mlflow.start_run():

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        "learning_rate": 0.26716716970531285,
        "max_depth": 9,
        "min_child_weight": 4.2386416350260365,
        "objective": "reg:linear",
        "reg_alpha": 0.14223944345839543,
        "reg_lambda": 0.35981469382798437,
        "seed": 42,
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, "validation")],
        early_stopping_rounds=50,
    )

    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
