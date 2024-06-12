from kedro.pipeline import pipeline, Pipeline
from typing import Dict
from .pipelines import (
    data_ingestion,
    data_split,
    data_cleaning
)


def register_pipelines() -> Dict[str, Pipeline]:
    
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    data_cleaning_pipeline = data_cleaning.create_pipeline()

    return {
        'data_ingestion': data_ingestion_pipeline,
        'data_split': data_split_pipeline,
        'data_preprocessing': data_cleaning_pipeline,
        '__default__': data_ingestion_pipeline + data_split_pipeline + data_cleaning_pipeline,
    }
