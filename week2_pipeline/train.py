import os
import pickle
import click

import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
@click.option(
    "--mlflow_tracking_uri",
    default="http://localhost:5000",
    help="Tracking URI of the MLflow server",
)
@click.option(
    "--mlflow_experiment_name",
    default="week2_homework",
    help="Name of the experiment",
)
def run_train(data_path: str, mlflow_tracking_uri: str, mlflow_experiment_name: str):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    train_data_path = os.path.join(data_path, "train.pkl")
    val_data_path = os.path.join(data_path, "val.pkl")

    X_train, y_train = load_pickle(train_data_path)
    X_val, y_val = load_pickle(val_data_path)

    mlflow.sklearn.autolog()

    rf = RandomForestRegressor(max_depth=10, random_state=0)

    with mlflow.start_run():

        mlflow.log_param("train-data-path", train_data_path)
        mlflow.log_param("val-data-path", val_data_path)

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":

    run_train()
