from kedro.pipeline import Pipeline, node
from .nodes import select_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                select_model,
                inputs=dict(
                    X_train="X_train_selected",
                    y_train="y_train_cleaned",
                    X_val="X_val_selected",
                    y_val="y_val_cleaned",
                    X_test="X_test_selected",
                    y_test="y_test_cleaned",
                    n_trials="params:n_trials",
                    # champion_model="champion_model",
                ),
                outputs="champion_model",
                name= "model_selection_node"
            )
        ]
    )