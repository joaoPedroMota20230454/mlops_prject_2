from kedro.pipeline import pipeline, Pipeline
from typing import Dict
from .pipelines import (
    data_unit_tests,
    data_split,
    data_cleaning,
    feature_engineering,
    feature_selection,
    model_selection,
    model_evaluation,
    data_drift,
)


def register_pipelines() -> Dict[str, Pipeline]:

    data_unit_tests_pipeline = data_unit_tests.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    data_cleaning_pipeline = data_cleaning.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    model_evaluation_pipeline = model_evaluation.create_pipeline()
    data_drift_pipeline = data_drift.create_pipeline()
    return {
        # common pipelines
        "data_unit_tests": data_unit_tests_pipeline,    
        "data_split": data_split_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_selection": model_selection_pipeline,
        
        "model_evaluation": model_evaluation_pipeline,
        "data_drift": data_drift_pipeline,

        # development pipeline
        "development": data_unit_tests_pipeline + \
            data_split_pipeline + \
                data_cleaning_pipeline + \
                    feature_engineering_pipeline + \
                        feature_selection_pipeline,
        
        # production pipeline
        # "production": ''
        
        # "__default__": ""  # kedro run (by default will run the __default__ pipeline)
        
    }
