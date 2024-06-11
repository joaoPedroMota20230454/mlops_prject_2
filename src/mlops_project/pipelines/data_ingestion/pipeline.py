from kedro.pipeline import Pipeline, node, pipeline
from .nodes import import_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= import_raw_data,
                inputs="params:raw_data_filepath",
                outputs= "raw_data",
                name="ingestion_node",
            ),

        ]
    )
