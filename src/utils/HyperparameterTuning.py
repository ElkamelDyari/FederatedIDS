from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_hyperparameter_space(model_name):
    """
        This function generates a hyperparameter space dictionary for the provided model name.

        It currently supports four types of models:
        "DecisionTree", "RandomForest", "ExtraTrees", and "XGBoost".
        Each model has its own parameter space structured for hyperparameter tuning.

        Parameters:
        model_name (str): The name of the model for which the hyperparameter space will be generated.

        Returns:
        dict: A dictionary containing the hyperparameter space for the specified model.

        Raises:
        ValueError: If the provided model_name is not supported.
    """

    if model_name == "DecisionTree":
        return {
            'max_depth': hp.quniform('max_depth', 5, 50, 1),
            'max_features': hp.quniform('max_features', 1, 21, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])
        }
    elif model_name == "RandomForest":
        return {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
            'max_depth': hp.quniform('max_depth', 5, 50, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])
        }
    elif model_name == "ExtraTrees":
        return {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
            'max_depth': hp.quniform('max_depth', 5, 50, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])
        }
    elif model_name == "XGBoost":
        return {
            'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
            'max_depth': hp.quniform('max_depth', 4, 100, 1),
            'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
        }

def objective_function(params, model_name, X_train, y_train):
    """
        This function defines the objective function for hyperparameter tuning.

        It uses the provided model name, hyperparameters, and training data to create a model,
        perform cross-validation, and calculate the F1 score.

        Parameters:
        params (dict): A dictionary containing hyperparameters for the model.
        model_name (str): The name of the model to be used.
        X_train (array-like): The training data (features) for the model.
        y_train (array-like): The training labels associated with the training data.

        Returns:
        dict: A dictionary containing the loss value (which is the negative of the calculated F1 score)
              and the status of the optimization process.
    """
    model = create_model(params, model_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(
        model, X_train, y_train,
        cv=skf,
        scoring=make_scorer(f1_score, average='weighted')
    ).mean()
    return {'loss': -f1, 'status': STATUS_OK}

def create_model(params, model_name):
    """
        This function creates a model based on the provided model name and parameters.

        It maps the model name to the corresponding model class and instantiates it
        with the given parameters. It currently supports four types of models:
        "DecisionTree", "RandomForest", "ExtraTrees", and "XGBoost".

        Parameters:
        params (dict): A dictionary of hyperparameters for the model.
        model_name (str): The name of the model to be created.

        Returns:
        A model instance of the specified model class, initialized with provided hyperparameters.

        Raises:
        Exception: If the provided model_name does not match any of the supported model types.
    """

    # Convert float params to int where needed
    params = {
        k: int(v) if isinstance(v, float) and k not in ['learning_rate', 'criterion'] else abs(float(v)) if k == 'learning_rate' else v
        for k, v in params.items()
    }

    # Decode categorical parameters if encoded as integers
    if 'criterion' in params and isinstance(params['criterion'], int):
        params['criterion'] = ['gini', 'entropy', 'log_loss'][params['criterion']]

    # Return model based on its name
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
