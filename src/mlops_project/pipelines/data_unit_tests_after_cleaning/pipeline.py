from kedro.pipeline import Pipeline, node
from .nodes import test_data_after_cleaning


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=test_data_after_cleaning,
                inputs=dict(
                    df='X_train_cleaned',
                    datasource_name='params:clean_datasource_name',
                    suite_name='params:clean_suite_name',
                    data_asset_name='params:X_train_cleaned_data_asset_name',
                    build_data_docs='params:build_data_docs',
                ),
                outputs="X_train_cleaned_validated",  # for now we are overwriting the data
                name="X_train_data_unit_tests_node",
            ),
            node(
                func=test_data_after_cleaning,
                inputs=dict(
                    df='X_test_cleaned',
                    datasource_name='params:clean_datasource_name',
                    suite_name='params:clean_suite_name',
                    data_asset_name='params:X_test_cleaned_data_asset_name',
                    build_data_docs='params:build_data_docs',
                ),
                outputs="X_test_cleaned_validated",  # # for now we are overwriting the data
                name="X_test_data_unit_tests_node",
            ),
        ]
    )

