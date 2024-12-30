import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.utils.MlflowUtils import get_best_model_params
from src.utils.Evaluate import evaluate_model
from src.utils.DataUtils import load_data


def create_model_from_params(Experiment_name="Hyperparameter Optimization"):
    """
        The function retrieves the best model and its parameters from a specified experiment, creates a model
        with these parameters, and returns it.

        The function performs the following steps:

        1. Fetch the best model and its parameters from the specified experiment.
        2. Remove the last item from the parameters dictionary.
        3. Decodes the 'criterion' parameter if it is encoded as an integer.
        4. Corrects any type mismatch in 'max_depth' and 'n_estimators' parameters by converting them into integers.
        5. Instantiate a model instance based on its name with the optimized parameters.

        Parameters:
        Experiment_name (str, optional): The name of the experiment from which to fetch the best model and its parameters.

        Returns:
        model: A model instance based on the best model's name. If model name isn't recognized, an error might occur.
    """

    # Fetch best model and its parameters
    best_model_info = get_best_model_params(Experiment_name)
    model_name = best_model_info["model_name"]
    run_id = best_model_info["run_id"]
    params = best_model_info["params"]
    params.popitem()
    # Load the best model from MLflow
    print(f"Loaded model {best_model_info['model_name']} \nfrom run ID: {run_id} \nwith params: {params}")

    # Decode categorical parameters if encoded as integers
    if 'criterion' in params and isinstance(params['criterion'], int):
        params['criterion'] = ['gini', 'entropy', 'log_loss'][params['criterion']]

    # Fix any parameter type mismatches
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])

    # Return model based on its name
    if model_name == "DecisionTree":
        model = DecisionTreeClassifier(**params)
    elif model_name == "RandomForest":
        model =  RandomForestClassifier(**params)
    elif model_name == "ExtraTrees":
        model =  ExtraTreesClassifier(**params)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)

    return model


def register_global_model(model, experiment_name, signature=None, input_example=None):
    """
        This function registers the provided model into the MLflow model registry.

        It performs the following steps:
        1. Sets the MLflow experiment to the provided experiment name.
        2. Logs the model in MLflow with the provided signature and input example.
        3. If a registered model with the same experiment name already exists, it uses the
           existing model, otherwise it creates a new registered model.
        4. Registers the new version of the model.

        Parameters:
        model : The trained model instance.
        experiment_name (str): The name of the registered model.
        signature: The model schema, if any (default is None).
        input_example: The example of input that this model can take, if any (default is None).

        Returns:
        None. The function operates by registering the model into MLflow model registry and
        printing out relevant steps.
    """

    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    # Log the model and get its URI
    with mlflow.start_run(run_name="Register_Global_Model"):

        # Log the model
        mlflow.sklearn.log_model(model,
                                 artifact_path="model",
                                 signature=signature,
                                 input_example=input_example,
                                 registered_model_name=experiment_name)
        model_uri = mlflow.get_artifact_uri("model")

    print(f"Model logged at URI: {model_uri}")

    # Check if the registered model already exists
    try:
        registered_model = client.get_registered_model(experiment_name)
        print(f"Model '{experiment_name}' already exists. Using existing model.")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"Creating new registered model: {experiment_name}")
            client.create_registered_model(experiment_name)
        else:
            print(f"Error checking registered model: {e}")
            raise

    # Register the new version of the model
    model_version = client.create_model_version(
        name=experiment_name,
        source=model_uri,
        run_id=None,  # Pass run_id if applicable
    )


    print(f"Registered new version '{model_version.version}' of model '{experiment_name}'")



def train_local_model(model, X_train, y_train):
    """
        This function fits the provided model using the provided training data and labels.

        Parameters:
        model : The model instance to be trained.
        X_train (array-like): The training data to fit the model.
        y_train (array-like): The labels for the training data.

        Returns:
        model : The provided model after being fit on X_train and y_train.
    """

    model.fit(X_train, y_train)
    return model

def aggregate_predictions(predictions, weights):
    """
        This function aggregates predictions from multiple clients using weighted averaging.

        Parameters:
        predictions (list): A list of arrays containing predictions from each client.
        weights (list): A list of weights corresponding to each client's predictions.

        Returns:
        aggregated (array): An array of aggregated predictions.
    """

    aggregated = np.average(predictions, axis=0, weights=weights)
    return np.round(aggregated).astype(int)

class FederatedLearningApp:

    def __init__(self, global_model, client_data_paths, rounds=1): # Add rounds parameter 3
        self.global_model = global_model
        self.client_data_paths = client_data_paths
        self.rounds = rounds

    def run(self):
        clients = []

        # Load and split data for each client
        for client_path in self.client_data_paths:
            print("Loading data from", client_path)
            X_train, X_val, y_train, y_val = load_data(client_path)
            clients.append({"train": (X_train, y_train), "val": (X_val, y_val)})

        for round_num in range(self.rounds):
            print(f"\nStarting Round {round_num + 1}")

            local_models = []
            local_f1s = []
            client_predictions = []
            client_weights = []
            i = 1
            # Train on each client's data
            for client in clients:

                X_train, y_train = client["train"]
                X_val, y_val = client["val"]

                # Train local model
                local_model = train_local_model(self.global_model, X_train, y_train)
                predictions = local_model.predict(X_val)
                # Evaluate local model
                metrics = evaluate_model(y_val, predictions)
                f1 = metrics["f1_score"]
                local_f1s.append(f1)
                local_models.append(local_model)

                # Collect predictions and weights for aggregation
                client_predictions.append(predictions)
                client_weights.append(len(y_train))

                # Log metrics to MLflow
                mlflow.set_experiment("Federated Training")
                with mlflow.start_run(run_name="Client_" + str(i)):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Create input example and infer signature
                    input_example = X_val.head(1)
                    signature = infer_signature(X_val, predictions)

                    mlflow.sklearn.log_model(local_model, "model", signature=signature, input_example=input_example)
                i+=1
            # Aggregate predictions
            aggregated_predictions = aggregate_predictions(client_predictions, client_weights)
            global_accuracy = accuracy_score(y_val, aggregated_predictions)

            print(f"Global accuracy after round {round_num + 1}: {global_accuracy}")

            # Update global model as the best performing model
            self.global_model = local_models[np.argmax(local_f1s)]

            print(f"Round {round_num + 1} completed.")


        return self.global_model, input_example, signature

