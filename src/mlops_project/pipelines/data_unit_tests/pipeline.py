from kedro.pipeline import Pipeline, node
from .nodes import test_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=test_data,
                inputs="raw_data",
                outputs="validated_data",
                name="data_unit_tests_node",
            ),
        ]
    )
