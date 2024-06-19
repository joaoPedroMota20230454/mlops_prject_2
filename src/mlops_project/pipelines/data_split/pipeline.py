from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=dict(
                    df="validated_data",
                    target_label="params:target_label",
                    set_sizes="params:set_sizes",
                    stratify="params:stratify",
                    random_state="params:random_state",
                ),
                outputs=[
                    "X_train", "X_val", "X_test",
                    "y_train", "y_val", "y_test"
                ],
                name="split_node",
            ),
        ]
    )
