"""
This is a boilerplate test file for pipeline 'model_selection'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
 
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator
from src.mlops_project.pipelines.model_selection.nodes import select_model
import os





def test_model_selection():
    train_sample_filepath = os.path.join("tests\sample\sample_model_input_train.csv")
    val_sample_filepath = os.path.join("tests\sample\sample_mode_input_val.csv")
    test_sample_filepath = os.path.join("tests\sample\sample_model_input_test.csv") 

    y_sample_train_filepath = os.path.join("tests\sample\sample_y_train.csv")
    y_sample_val_filepath = os.path.join("tests\sample\sample_y_val.csv")
    y_sample_test_filepath = os.path.join("tests\sample\sample_y_test.csv")


    # dfs
    train_sample = pd.read_csv(train_sample_filepath)
    val_sample = pd.read_csv(val_sample_filepath)
    test_sample = pd.read_csv(test_sample_filepath)

    y_sample_train = pd.read_csv(y_sample_train_filepath)
    y_sample_val = pd.read_csv(y_sample_val_filepath)
    y_sample_test = pd.read_csv(y_sample_test_filepath)


    best_model = select_model(
                            train_sample,
                            y_sample_test,
                            val_sample,
                            y_sample_val,
                            test_sample,
                            y_sample_test,
                            )
    assert isinstance(best_model, BaseEstimator)

    