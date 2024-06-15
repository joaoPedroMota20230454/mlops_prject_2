from kedro.pipeline import pipeline, Pipeline
from typing import Dict
from .pipelines import (
    data_unit_tests,
    data_split,
    data_cleaning,
    feature_engineering,
)


def register_pipelines() -> Dict[str, Pipeline]:

    data_unit_tests_pipeline = data_unit_tests.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    data_cleaning_pipeline = data_cleaning.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()

    return {
        "data_unit_tests": data_unit_tests_pipeline,    
        "data_split": data_split_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        
        "__default__": data_unit_tests_pipeline + \
            data_split_pipeline + \
                data_cleaning_pipeline + \
                    feature_engineering_pipeline
    }
