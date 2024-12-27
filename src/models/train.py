import pandas as pd
from src.utils import ImbalanceHandling
from src.data import DataCleaning
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

def split(path):
    data = pd.read_csv(path)
    data = DataCleaning.data_cleaning(data)
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train_smote, y_train_smote = ImbalanceHandling.over_sample(X_train, y_train)
    return X_train, X_train_smote, X_test, y_train, y_train_smote, y_test

# Define hyperparameter optimization space for each model
def get_search_space(model_name, path):
    if model_name == "DecisionTree":
        return {
            'max_depth': hp.quniform('max_depth', 5, 50, 1),
            'max_features': hp.quniform('max_features', 1, X_train.shape[1], 1),
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
            'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
            'max_depth': hp.quniform('max_depth', 5, 50, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
        }

# Define model initialization
def initialize_model(params, model_name):
    params = {k: int(v) if isinstance(v, float) and k != 'learning_rate' else v for k, v in params.items()}
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)

# Objective function for hyperparameter optimization
def objective(params, model_name):
    model = initialize_model(params, model_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(
        model, X_train, y_train,
        cv=skf,
        scoring=make_scorer(f1_score, average='weighted')
    ).mean()
    return {'loss': -f1, 'status': STATUS_OK}

path ="../../data/samples/data1.csv"

# Load and split dataset
X_train, X_train_smote, X_test, y_train, y_train_smote, y_test = split(path)

# List of models to optimize
models = ["DecisionTree", "RandomForest", "ExtraTrees", "XGBoost"]

# MLflow tracking
mlflow.set_experiment("Hyperparameter Optimization")

for model_name in models:
    with mlflow.start_run(run_name=model_name):
        space = get_search_space(model_name)

        # Hyperparameter optimization
        trials = Trials()
        best_params = fmin(
            fn=lambda params: objective(params, model_name),
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        # Train the final model with the best parameters
        best_model = initialize_model(best_params, model_name)
        best_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log results to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"{model_name}: Best F1 Score = {f1:.4f}")



