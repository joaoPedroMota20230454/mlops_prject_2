from kedro.pipeline import Pipeline, node
from .nodes import select_features, filter_dataset

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                select_features,
                inputs=dict(
                    X_train="X_train_featurized",
                    y_train="y_train_cleaned",
                    feature_selection="params:feature_selection",
                    model_params="params:fs_model_params",
                    n_features="params:n_features",
                    manual_features="params:manual_features",
                ),
                outputs="selected_features",
                name= "feature_selection_node"
            ),
            node(
                filter_dataset,
                inputs=["X_train_featurized", "selected_features"],
                outputs="X_train_selected",
                name="X_train_selected_node",
            ),
            node(
                filter_dataset,
                inputs=["X_val_featurized", "selected_features"],
                outputs="X_val_selected",
                name="X_val_selected_node",
            ),
            node(
                filter_dataset,
                inputs=["X_test_featurized", "selected_features"],
                outputs="X_test_selected",
                name="X_test_selected_node",
            ),
        ]
    )
