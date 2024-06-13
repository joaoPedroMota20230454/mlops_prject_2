from kedro.pipeline import Pipeline, node
from .nodes import test_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=test_data,
                inputs=dict(
                    df='raw_data',
                    build_data_docs='params:build_data_docs'
                ),
                outputs="validated_data",
                name="data_unit_tests_node",
            ),
        ]
    )
