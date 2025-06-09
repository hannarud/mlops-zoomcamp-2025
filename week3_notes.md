# Week 3 notes

## Turning pipeline into a script

After the pipeline is ready in a script (see [.py file](week3_duration-prediction.py)), it is possible to call the pipeline by:
1. Starting MLFlow: `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --default-artifact-root ./artifacts`
2. Running the script: `python week3_duration-prediction.py --year 2023 --month 1`

It will create a model and run it with MLFlow + save the run ID in `run_id.txt`.

Next steps to create a pipeline with Prefect:

## Prefect

### Installation

`pip install prefect`

### Basic workflow

Start a local server:
* `prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`
* `prefect server start`

Then it is possible to go to the dashboard at http://127.0.0.1:4200/dashboard.

Assuming that MLFlow is also running, it is now possible to run the script and observe the results in Prefect as well as in the console.

### Deployment

`prefect project init` - this is not working, the correct command is `prefect init`. This initializes a Prefect project in the current directory.

Go to Prefect UI and create a new work pool. The type to select is "Process", since it's the most basic option.

Deploy the flow with the following command: `prefect deploy week3_duration-prediction.py:main_flow -n nyc-taxi-flow -p "MLOps Zoomcamp Week 3"`.

To execute flow runs, we'll need to start a worker: `prefect worker start --pool "MLOps Zoomcamp Week 3"`.

And then it is possible to run the deployment either from the command line (`prefect deployment run) or from the UI.

**VERY IMPORTANT**: since Prefect involves git and clones a copy of the repo with a current state from the remote, it is very important to first `git push` all the changes!

### Scheduling

Scheduling is easily available from the UI, and there are CLI commands as well.
