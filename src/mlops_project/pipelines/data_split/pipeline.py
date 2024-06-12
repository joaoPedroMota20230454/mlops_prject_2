from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_data,
                inputs=dict(
                    df = 'raw_data',
                    target = 'params:target_column',
                    test_size = 'params:test_size',
                    random_state = 'params:random_state'
                ),
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name='split_node',
            ),

        ]
    )
