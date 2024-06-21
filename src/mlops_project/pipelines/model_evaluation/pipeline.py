"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.4
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import model_predict, model_evaluation





def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            model_predict,
            inputs=["X_test_selected", "champion_model", "selected_features"],
            outputs=["df_with_predictions", "describe_servings"],
            name="model_predict_node",
        )
        node(
            model_evaluation,
            inputs=["df_with_predictions", "y_test_cleaned"],
            outputs="model_metrics_on_production",
            name="model_evaluation_node",
        )

    ])
