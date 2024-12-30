from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

def evaluate_model(y_true, predictions, proba_predictions=None):
    """
    Evaluates a classification model and computes various metrics.

    Parameters:
    - y_true: array-like, true labels.
    - predictions: array-like, predicted labels.
    - proba_predictions: array-like, predicted probabilities for the positive class (optional).

    Returns:
    - metrics: dict containing calculated metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, average='weighted'),  # Updated for multiclass
        "recall": recall_score(y_true, predictions, average='weighted'),        # Updated for multiclass
        "f1_score": f1_score(y_true, predictions, average='weighted'),          # Ensuring F1 is included
    }

    # Precision-Recall AUC
    if proba_predictions is not None and len(set(y_true)) == 2:  # Only compute PR AUC for binary classification
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba_predictions)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    else:
        metrics["pr_auc"] = 0

    return metrics
