# Week 3 notes

After the pipeline is ready in a script (see [.py file](week3_duration-prediction.py)), it is possible to call the pipeline by:
1. Starting MLFlow: `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --default-artifact-root ./artifacts`
2. Running the script: `python week3_duration-prediction.py --year 2023 --month 1`

It will create a model and run it with MLFlow + save the run ID in `run_id.txt`.

Next steps to create a pipeline with Prefect:
