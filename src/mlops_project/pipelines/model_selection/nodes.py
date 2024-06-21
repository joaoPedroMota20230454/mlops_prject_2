from typing import Tuple
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator

import optuna
import mlflow
from mlflow.tracking import MlflowClient

# from mlops_project.pipelines.utils import load_registered_model_version  # import error fix this


# should this also go into a 'utils'-like module?
MODELS_DICT = {
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'tune': True
        },
    
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'tune': True
    },
}


def register_model(
    model_path: str,
    model_name: str,
    model_tag: str = None,
    model_version: int = None,
    model_alias: str = None
) -> None:
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        model_path (str): The path to the model to register.
        model_name (str): The name of the model to register.
        model_tag (str): A tag to add to the model.
        model_version (int): The version of the model to register.
        model_alias (str): An alias to add to the model.
    """
    mlflow.register_model(
        model_path, model_name
    )
    
    if any(var is not None for var in [model_tag, model_version, model_alias]):
        client = MlflowClient()
    
        if model_tag:
            client.set_registered_model_tag(
                model_name, "task", "classification"
            )
        
        if model_version:
            client.set_model_version_tag(
                model_name, str(model_version), "validation_status", "approved"
            )
        
        if model_alias:
            client.set_registered_model_alias(
                model_name, model_alias, str(model_version)
            )


def delete_registered_model(model_name: str) -> None:
    """
    Delete a model from the MLflow Model Registry. If 'all' is
    passed as the model_name, all registered models will be deleted.
    
    Args:
        model_name (str): The name of the model to delete.
    """
    
    client = MlflowClient()
    
    if model_name == "all":
        # get names of registered models
        registered_models = client.list_registered_models()
        
        # delete all registered models
        for model in registered_models:
            client.delete_registered_model(name=model.name)
    
    else:
        client.delete_registered_model(name=model_name)


def update_model_alias():
    pass


# NOTE: if true champion model is stored locally then this probably won't be necessary
def load_registered_model_version(model_name: str, version: int):
    """
    Load a specific version of a registered model from the
    MLflow Model Registry. If version is -1, load the latest version.
    
    Args:
        model_name (str): The name of the registered model.
        version (int): The version of the model to load, or -1 for the latest version.
    
    Returns:
        model: The loaded model.
    """
    client = MlflowClient()

    if version == -1:
        # get the latest version
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise Exception(f"No versions found for model {model_name}")
        latest_version = max(versions, key=lambda v: v.version).version
        version = latest_version
    
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise Exception(f"Could not load model from Registry: {e}")


def optuna_objective(
    trial: optuna.Trial,
    train: Tuple[pd.DataFrame, pd.Series],
    val: Tuple[pd.DataFrame, pd.Series],
    model_name: str
) -> float:
    
    model = MODELS_DICT[model_name]['model']
    
    X_train, y_train = train
    X_val, y_val = val
    
    # unfortunately there really is not better way than if-else statements
    if model_name == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_categorical('max_depth', [None, 5, 10])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        
        model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight
        )
    
    elif model_name == 'GradientBoostingClassifier':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1)
        max_depth = trial.suggest_categorical('max_depth', [3, 5, 10])
        
        model.set_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
    
    else:
        raise ValueError(f"Model {model_name} not found in MODELS_DICT")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, pos_label=1) # f1-score for the pos. class (i.e. readmitted)
    
    return score


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 50,
    champion_model: BaseEstimator = None,
    models: dict = MODELS_DICT
) -> BaseEstimator:
    
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
        
    best_model = None
    best_score = 0
    best_run_id = None
    best_model_name = None
    
    mlflow.set_experiment("model_selection")
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)
    
    with mlflow.start_run(run_name="model_selection_run", nested=True):
        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=model_name, nested=True) as run:
                
                print(model_name, run.info.run_id)
                
                if model_info['tune']:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(
                        lambda trial: optuna_objective(
                            trial,
                            (X_train, np.ravel(y_train)),
                            (X_val, np.ravel(y_val)),
                            model_name
                        ),
                        n_trials=n_trials
                    )
                    model = model_info['model'].set_params(**study.best_params)
                else:
                    model = model_info['model']
                
                model.fit(X_combined, np.ravel(y_combined))
                train_pred = model.predict(X_combined)
                test_pred = model.predict(X_test)
                
                train_report = classification_report(np.ravel(y_combined), train_pred, output_dict=True)
                test_report = classification_report(np.ravel(y_test), test_pred, output_dict=True)
                
                overfitting = np.abs(train_report["1"]["f1-score"] - test_report["1"]["f1-score"])
                
                # updating best model
                if (overfitting < .1) & (test_report["1"]["f1-score"] > best_score):
                    best_model = model
                    best_score = test_report["1"]["f1-score"]
                    best_run_id = run.info.run_id
                    best_model_name = model_name

                # NOTE: not all items are logged, so we are adding more
                # metrics logging (train and test)
                mlflow.log_metric("training_precision_1", train_report["1"]["precision"])
                mlflow.log_metric("training_recall_1", train_report["1"]["recall"])
                mlflow.log_metric("training_f1_score_1", train_report["1"]["f1-score"])
                mlflow.log_metric("training_accuracy", train_report["accuracy"])
                
                mlflow.log_metric("testing_precision_1", test_report["1"]["precision"])
                mlflow.log_metric("testing_recall_1", test_report["1"]["recall"])
                mlflow.log_metric("testing_f1_score_1", test_report["1"]["f1-score"])
                mlflow.log_metric("testing_accuracy", test_report["accuracy"])
        
        # if models show great overfitting then we don't register any model
        if best_model is not None:
            # comparing challenger to champion
            if champion_model is not None:
                # check this out: https://mlflow.org/docs/latest/model-registry.html
                
                # TODO: in order to have only one model registered at a time
                # should we delete all previously registered models?
                # or should we update the alias of the other models to 'challenger'
                # and name the best model as champion?
                
                champion_test_pred = champion_model.predict(X_test)
                champion_f1 = f1_score(np.ravel(y_test), champion_test_pred, pos_label=1)
                if best_score > champion_f1:
                    register_model(
                        f"runs:/{best_run_id}/model",
                        best_model_name,
                        model_version=1,  # if v1 is already registered, new version will be created
                        model_alias="champion"
                    )
            else:
                # if there is not champion model we register the best model
                register_model(
                    f"runs:/{best_run_id}/model",
                    best_model_name,
                    model_version=1,  # if v1 is already registered, new version will be created
                    model_alias="champion"
                )
            
    return best_model


if __name__ == '__main__':
    # testing
    import os

    params = {
        'X_train': pd.read_csv(os.path.join('data', '06_model_input', 'X_train_selected.csv')),
        'y_train': pd.read_csv(os.path.join('data', '04_clean', 'y_train_cleaned.csv')),
        'X_val': pd.read_csv(os.path.join('data', '06_model_input', 'X_val_selected.csv')),
        'y_val': pd.read_csv(os.path.join('data', '04_clean', 'y_val_cleaned.csv')),
        'X_test': pd.read_csv(os.path.join('data', '06_model_input', 'X_test_selected.csv')),
        'y_test': pd.read_csv(os.path.join('data', '04_clean', 'y_test_cleaned.csv')),
        'n_trials': 1,
        'champion_model': None,
        'models': MODELS_DICT
    }
    
    select_model(**params)
    
    # i think the weird error im getting is because when i start a pipeline run
    # kedro tracks the pipeline run id and then when i start a nested run
    # i get that an active run is already in place, i.e. at the same time
    # kedro is tracking model_selection pipeline and the experiment
