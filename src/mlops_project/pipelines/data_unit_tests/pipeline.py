from kedro.pipeline import Pipeline, node
from .nodes import test_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=test_data,
                inputs=dict(
                    df='raw_data',
                    datasource_name='params:raw_datasource_name',
                    suite_name='params:raw_suite_name',
                    data_asset_name='params:raw_data_asset_name',
                    build_data_docs='params:build_data_docs'
                ),
                outputs="validated_data",
                name="data_unit_tests_node",
            ),
        ]
    )
