from .nodes import data_drift_monitoring
from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                data_drift_monitoring,
                inputs=["X_train_cleaned", "X_test_cleaned"],
                outputs=None,
                name="data_drift_monitoring_node",
            )
        ]
    )