from kedro.pipeline import Pipeline, node
from .nodes import clean_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_data,
                inputs="raw_data",
                outputs="clean_data",
                name="clean_data_node",
            )
        ]
    )
