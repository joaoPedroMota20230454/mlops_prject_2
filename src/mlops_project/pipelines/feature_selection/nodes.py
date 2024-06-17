import os
import json
import pickle
from typing import Any, Dict, Tuple


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import mlflow


def best_n_features(X_train:pd.DataFrame, y_train:pd.Series, parameters:Dict[str,Any]):
    """
    Perform feature selection using Recursive Feature Elimination (RFE).

    Args:
    --
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    n_features (int): Number of features to select.

    Returns:
    --
    X_train_selected (pd.DataFrame): Training data with selected features.
    selected_features (list): List of selected feature names.
    """
    if parameters["feature_selection"] == "rfe":
        try:
            with open(os.path.join("..\\data\\06_models\\champion_model.pkl", "rb")) as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters["baseline_model_params"])
        rfe = RFE(classifier, n_features_to_select=parameters["nr_features"])
        rfe = rfe.fit(X_train, np.ravel(y_train))
        most_important_features = rfe.get_support(1)
        X_cols = X_train.columns[most_important_features].tolist()
    return X_cols