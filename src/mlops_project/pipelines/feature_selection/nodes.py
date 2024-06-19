from typing import Any, Dict, Union

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_selection: str,
    model_params: Union[Dict[str, Any], None],
    n_features: Union[int, None],
    manual_features: list,
    verbose: int = 0,
) -> list:
    """
    Perform feature selection on the training data. Can be done using
    RFE with a RF Classifier, using all features or even just passing
    a pre-defined list of features.

    Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Training labels.
        feature_selection (str): Method for feature selection. Can be "rfe", "all", "manual".
        model_params (Dict[str, Any]): Parameters for the RF Classifier. Can be None if "all" or "manual".
        n_features (int): Number of features to select. Can be None if "all" or "manual".
        manual_features (list): List of manual features to use, only applicable if "manual".
        verbose (int): Verbosity level.

    Returns:
        selected_features (list): List of selected feature names.
    """
    if feature_selection == "rfe":
        classifier = RandomForestClassifier(**model_params)    
        rfe = RFE(classifier, n_features_to_select=n_features)
        rfe = rfe.fit(X_train, np.ravel(y_train))
        most_important_features = rfe.get_support(1)
        X_cols = X_train.columns[most_important_features].tolist()
    
    elif feature_selection == "all":
        X_cols = X_train.columns.tolist()
    
    elif feature_selection == "manual":
        
        if not isinstance(manual_features, list):
            raise ValueError("Manual features must be a list")

        if len(manual_features) == 0:
            raise ValueError("Manual features list is empty")
        
        X_cols = manual_features
    
    elif isinstance(feature_selection, list):
        X_cols = feature_selection
    
    else:
        raise ValueError(f"Feature selection method {feature_selection} not supported")  
    
    if verbose > 0:
        print(f"Selected features: {X_cols}")

    return X_cols


def filter_dataset(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Filter the dataset to keep only the selected features.
    
    Args:
        df (pd.DataFrame): Data to filter.
        features (list): List of features to keep.
    
    Returns:
        pd.DataFrame: Filtered data.
    """
    return df[features]
