import os
import sys
import mlflow
import dagshub
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.MlflowUtils import  run_hyperparameter_optimization


# Initialize DagsHub MLflow
mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/FederatedIDS.mlflow")
dagshub.init(repo_owner='ElkamelDyari', repo_name='FederatedIDS', mlflow=True)

# Define the path to the data
file_path = os.path.join(os.path.dirname(__file__), '../../data/samples/data1.csv')

# Run hyperparameter optimization
run_hyperparameter_optimization(file_path, use_smote=False)
