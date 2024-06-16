import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pickle
import mlflow
from typing import Any, Dict, Tuple
import numpy as np
import json
import os



# def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, n_features=12,):
#     """
#     Perform feature selection using Recursive Feature Elimination (RFE).

#     Args:
#     --
#     X_train (pd.DataFrame): Training features.
#     y_train (pd.Series): Training target.
#     n_features (int): Number of features to select.

#     Returns:
#     --
#     X_train_selected (pd.DataFrame): Training data with selected features.
#     selected_features (list): List of selected feature names.
#     """
#     with mlflow.start_run():
#         model = RandomForestClassifier()
#         rfe= RFE(estimator=model, n_features_to_select=n_features)
#         rfe.fit(X_train, y_train)

#         selected_features=X_train.columns[rfe.support_].tolist()
#         X_train_selected=X_train[selected_features]


#         mlflow.log_param("selected_features", selected_features)
#         mlflow.log_param("number_features", n_features)

#         return X_train_selected, selected_features
    



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