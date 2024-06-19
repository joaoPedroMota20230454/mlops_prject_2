from kedro.pipeline import Pipeline, node
from .nodes import add_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                add_features,
                inputs=dict(
                    df="X_train_cleaned",
                    upload_to_feature_store="params:upload_to_feature_store",
                ),
                outputs="X_train_featurized",
                name="X_train_feature_engineering_node",
            ),
            node(
                add_features,
                inputs=dict(
                    df="X_val_cleaned",
                    upload_to_feature_store="params:upload_to_feature_store",
                ),
                outputs="X_val_featurized",
                name="X_val_feature_engineering_node",
            ),
            node(
                add_features,
                inputs=dict(
                    df="X_test_cleaned",
                    upload_to_feature_store="params:upload_to_feature_store",
                ),
                outputs="X_test_featurized",
                name="X_test_feature_engineering_node",
            ),
        ]
    )