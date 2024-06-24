from kedro.pipeline import Pipeline, node
from .nodes import make_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                make_predictions,
                inputs=["X", "params:predict_proba"],
                outputs="predictions",
                name= "model_prediction_node"
            )
        ]
    )