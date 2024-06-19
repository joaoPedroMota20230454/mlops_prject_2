# TODO
# load champion model if one exists

# optuna with sklearn integration: https://optuna.readthedocs.io/en/v3.3.0/reference/generated/optuna.integration.OptunaSearchCV.html

from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator

import optuna
import mlflow


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


def optuna_objective(
    trial: optuna.Trial,
    train: Tuple[pd.DataFrame, pd.Series],
    val: Tuple[pd.DataFrame, pd.Series],
    model_name: str
) -> float:
    
    model = MODELS_DICT[model_name]
    
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


def model_selection_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 100,
    champion_model: BaseEstimator = None,
    models: dict = MODELS_DICT
) -> BaseEstimator:
    
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    
    results = {}
    
    mlflow.set_experiment("model_selection") 
    
    for model_name, model_info in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            
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
            else:
                model = model_info['model']
            
            model.fit(X_combined, np.ravel(y_combined))
            y_pred = model.predict(X_test)
            report = classification_report(np.ravel(y_test), y_pred, output_dict=True)
            
            # model logging
            mlflow.sklearn.log_model(model, model_name)
            
            # parameters logging
            mlflow.log_params(model.get_params())
            
            # metrics logging
            mlflow.log_metric("precision_1", report["1"]["precision"])
            mlflow.log_metric("recall_1", report["1"]["recall"])
            mlflow.log_metric("f1_score_1", report["1"]["f1-score"])
            mlflow.log_metric("support_1", report["1"]["support"])
            mlflow.log_metric("accuracy", report["accuracy"])
            
            # also log features
            mlflow.log_param("features", X_combined.columns.tolist())
            
            # TODO check if model beats the current champion model
            # if so, save it as the new champion model

    
    return results  # or best model??


if __name__ == '__main__':
    # testing
    pass