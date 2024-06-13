from kedro.pipeline import pipeline, Pipeline
from typing import Dict
from .pipelines import (
    data_unit_tests,
    data_split,
    data_cleaning,
    data_unit_tests_after_cleaning
)


def register_pipelines() -> Dict[str, Pipeline]:

    data_unit_tests_pipeline = data_unit_tests.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    data_cleaning_pipeline = data_cleaning.create_pipeline()
    data_unit_tests_after_cleaning_pipeline = data_unit_tests_after_cleaning.create_pipeline()

    return {
        "data_unit_tests": data_unit_tests_pipeline,    
        "data_split": data_split_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "data_unit_tests_after_cleaning": data_unit_tests_after_cleaning_pipeline,
        "__default__": data_unit_tests_pipeline + \
            data_split_pipeline + \
                data_cleaning_pipeline + \
                    data_unit_tests_after_cleaning_pipeline
    }
