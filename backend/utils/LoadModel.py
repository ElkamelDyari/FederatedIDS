import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize DagsHub MLflow
tracking_uri = os.getenv('TRAKING_URI')
mlflow.set_tracking_uri(tracking_uri)
#dagshub.init(repo_owner='ElkamelDyari', repo_name='FederatedIDS', mlflow=True)


def retrieve_latest_registered_model(experiment_name):
    """
    Retrieve the latest version of a globally registered model from MLflow.

    Args:
        experiment_name (str): The name of the registered model.

    Returns:
        model: The latest version of the registered model, loaded as a Python object.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        latest_run = client.search_runs(experiment_ids=[experiment.experiment_id],
                                        order_by=["attribute.start_time DESC"],
                                        max_results=1)[0]
        latest_run_id = latest_run.info.run_id
        model_uri = f"runs:/{latest_run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        print(f"Latest Run ID: {latest_run_id}")
        return model
    else:
        print(f'No experiment with name {experiment_name} found.')
        return None

