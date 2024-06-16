from kedro.pipeline import Pipeline, node
from .nodes import best_n_features

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                best_n_features,
                # "n_features_to_select"
                inputs=["X_train_featurized","y_train_cleaned","parameters"],
                outputs="best_columns",
                name= "feature_selection_node"

            )
        ]
    )
