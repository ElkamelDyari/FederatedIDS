import sys
import os
import dagshub
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.FederatedUtils import register_global_model, create_model_from_params, FederatedLearningApp

# Initialize DagsHub MLflow
mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/FederatedIDS.mlflow")
dagshub.init(repo_owner='ElkamelDyari', repo_name='FederatedIDS', mlflow=True)

# Define the paths to the client data
CLIENT_DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '../../data/samples/data1.csv'),
    os.path.join(os.path.dirname(__file__), '../../data/samples/data2.csv'),
    os.path.join(os.path.dirname(__file__), '../../data/samples/data3.csv'),
    os.path.join(os.path.dirname(__file__), '../../data/samples/data4.csv')
]

# Initialize global model and start federated learning
model_mlflow = create_model_from_params()
federated_app = FederatedLearningApp(model_mlflow, CLIENT_DATA_PATHS)
model_mlflow, input_example, signature = federated_app.run()
register_global_model(model_mlflow, experiment_name="Federated Learning IDS", signature=signature, input_example=input_example)
print("Federated Learning completed successfully.")


