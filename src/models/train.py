import os
import sys
import mlflow
import dagshub
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.MlflowUtils import  run_hyperparameter_optimization


load_dotenv()
# Initialize DagsHub MLflow
tracking_uri = os.getenv('TRAKING_URI')
repo_owner = os.getenv('REPO_OWNER')
repo_name = os.getenv('REPO_NAME')

mlflow.set_tracking_uri(tracking_uri)

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# Define the path to the data
file_path = os.path.join(os.path.dirname(__file__), '../../data/samples/data1.csv')

# Run hyperparameter optimization
run_hyperparameter_optimization(file_path, use_smote=False)
