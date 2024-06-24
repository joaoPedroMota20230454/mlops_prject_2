from typing import Tuple
import warnings; warnings.filterwarnings('ignore')
import pickle

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

import mlflow
from mlflow.tracking import MlflowClient


def make_predictions(champion_model: pickle.Pickler, X: pd.DataFrame, predict_proba: bool) -> Tuple[np.ndarray]:
    """
    Make predictions using the champion model.
    
    Args:
        champion_model: The champion model.
        X: The input data.
        
    Returns:
        predictions: The predictions.
        probabilities: The probabilities.
    """
    
    if predict_proba:
        predictions = champion_model.predict_proba(X)
    else:
        predictions = champion_model.predict(X)
    
    return predictions
