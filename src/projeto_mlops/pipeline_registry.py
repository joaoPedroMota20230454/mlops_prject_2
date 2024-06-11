from kedro.pipeline import pipeline, Pipeline
from typing import Dict

from .pipelines import (
    data_cleaning as data_cleaning,
)


def register_pipelines() -> Dict[str, Pipeline]:
    data_cleaning_pipeline = data_cleaning.create_pipeline()

    return {
        "data_cleaning": data_cleaning_pipeline,
    }