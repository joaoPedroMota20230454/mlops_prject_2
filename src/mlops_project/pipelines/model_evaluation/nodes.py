"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.4
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  model: pickle.Pickler, columns: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        y_test (pd.DataFrame): True values.
        model (pickle): Trained model.
        columns (pickle): Columns used for prediction.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """
    # Predict
    y_pred = model.predict(X[columns])

    # Create dataframe with predictions
    X['y_pred'] = y_pred
    
    # Create dictionary with predictions
    describe_servings = X.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings




def model_evaluation(X:pd.Dataframe, y_true_values:pd.Dataframe) -> Dict[str, Any]:

    accuracy = accuracy_score(y_true_values, X['y_pred'])
    precision = precision_score(y_true_values, X['y_pred'])
    recall = recall_score(y_true_values, X['y_pred'])
    f1 = f1_score(y_true_values, X['y_pred'])
    roc_auc = roc_auc_score(y_true_values, X['y_pred'])
    conf_matrix = confusion_matrix(y_true_values, X['y_pred'])

    X_eldery = X[X['age'] > 60]
    y_eldery = y_true_values[y_true_values["encounter_id"] == X_eldery["encounter_id"]]

    accuracy_eldery = accuracy_score(y_eldery, X_eldery['y_pred'])
    precision_eldery = precision_score(y_eldery, X_eldery['y_pred'])
    recall_eldery = recall_score(y_eldery, X_eldery['y_pred'])
    f1_eldery = f1_score(y_eldery, X_eldery['y_pred'])
    roc_auc_eldery = roc_auc_score(y_eldery, X_eldery['y_pred'])
    conf_matrix_eldery = confusion_matrix(y_eldery, X_eldery['y_pred'])

    dict_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix,
        "accuracy_eldery": accuracy_eldery,
        "precision_eldery": precision_eldery,
        "recall_eldery": recall_eldery,
        "f1_eldery": f1_eldery,
        "roc_auc_eldery": roc_auc_eldery,
        "conf_matrix_eldery": conf_matrix_eldery
    }
    return dict_metrics

