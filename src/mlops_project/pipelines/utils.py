"""
Storing utility functions for the project
"""

import mlflow


def load_registered_model_version(model_name: str, version: int):
    """
    Load a specific version of a registered model from the
    MLflow Model Registry.
    
    Args:
        model_name (str): The name of the registered model.
        version (int): The version of the model to load.
    
    Returns:
        model: The loaded model.
    """
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise Exception(f"Could not load model from MLflow Model Registry: {e}")