from kedro.pipeline import Pipeline, node
from .nodes import clean_df


# since our data is split into train and test sets we'll need
# to clean them seperately
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_df,
                inputs=["X_train", "y_train"],
                outputs=["X_train_cleaned", "y_train_cleaned"],
                name="X_train_cleaning_node",
            ),
            node(
                clean_df,
                inputs=["X_test", "y_test"],
                outputs=["X_test_cleaned", "y_test_cleaned"],
                name="X_test_cleaning_node",
            )
        ]
    )
