import os
import time
import mlflow
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix

from src.utils.Evaluate import evaluate_model
from src.utils.DataUtils import load_and_split_data
from src.utils.HyperparameterTuning import get_hyperparameter_space, create_model, objective_function


def get_best_model_params(experiment_name, metric="f1_score"):
    """
    Retrieve the hyperparameters of the best model from an MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        metric (str): The metric to sort runs by, in descending order.

    Returns:
        dict: A dictionary containing the model name and its parameters.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Query the runs sorted by the metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    best_run = runs[0]
    model_name = best_run.data.params.get("model_name")
    params = {k: v for k, v in best_run.data.params.items() if k != "model_name"}

    # Convert numeric parameters back to their original types
    for key, value in params.items():
        try:
            params[key] = eval(value)
        except (SyntaxError, NameError):
            params[key] = value  # Keep as string if eval fails

    return {"model_name": model_name, "params": params, "run_id": best_run.info.run_id}



def run_hyperparameter_optimization(path, use_smote=False, models=None):
    """
        This function performs hyperparameter optimization for a given set of models on a dataset.

        The dataset is loaded and split, then hyperparameters are tuned and logged for each model
        using the fmin function from the Hyperopt library. The process is repeated for each
        model in the provided models list. The function logs the best parameters,
        model performance metrics, and additional information to MLFlow.

        Parameters:
        path (str): Path to the dataset file.
        use_smote (bool, optional): Whether to use SMOTE oversampling. Default is False.
        models (list, optional): List of model names to optimize.
                                 Default is ["ExtraTrees", "DecisionTree", "XGBoost"].

        Returns:
        None

        Side effects:
        Logs the following information to MLFlow for each model:
        - best parameters
        - model performance metrics (F1 score, etc.)
        - empirical confusion matrix of the model
        - trained model
        - some additional info such as training duration and the datatype
    """

    if models is None:
        models = ["ExtraTrees", "DecisionTree", "XGBoost"]

    X_train, X_test, y_train, y_test = load_and_split_data(path)
    X_train_final =  X_train
    y_train_final = y_train

    mlflow.set_experiment("Hyperparameter Optimization")

    for model_name in models:
        with mlflow.start_run(run_name=model_name):
            start_time = time.time()
            space = get_hyperparameter_space(model_name)
            trials = Trials()
            best_params = fmin(
                fn=lambda params: objective_function(params, model_name, X_train_final, y_train_final),
                space=space,
                algo=tpe.suggest,
                max_evals=5,  # Change this to 50 for better results
                trials=trials
            )
            if model_name != "XGBoost" and 'criterion' in best_params:
                best_params['criterion'] = 'gini' if best_params['criterion'] == 0 else 'entropy'
            if "learning_rate" in best_params:
                best_params['learning_rate'] = abs(best_params['learning_rate'])

            print("Best parameters:", best_params)
            best_model = create_model(best_params, model_name)
            best_model.fit(X_train_final, y_train_final)

            predictions = best_model.predict(X_test)
            proba_predictions = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            metrics = evaluate_model(y_test, predictions, proba_predictions)


            end_time = time.time()
            duration = end_time - start_time

            # Log essential items
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(best_params)
            mlflow.log_param("Data Type", "SMOTE" if use_smote else "Normal")

            # Create input example and infer signature
            input_example = X_train_final.head(1)
            signature = infer_signature(X_train_final, predictions)

            mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=input_example)

            # Log optional items for clarity
            mlflow.log_metric("Training Duration", duration)
            cm = confusion_matrix(y_test, predictions)
            confusion_matrix_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/confusion_matrix.csv')

            pd.DataFrame(cm).to_csv(confusion_matrix_path)

            # Log the csv file as an artifact
            mlflow.log_artifact(local_path=confusion_matrix_path, artifact_path="confusion_matrix.csv")

            f1 = metrics['f1_score']
            print(f"{model_name}: Best F1 Score = {f1:.4f}")



