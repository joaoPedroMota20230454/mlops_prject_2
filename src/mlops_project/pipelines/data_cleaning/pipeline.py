from kedro.pipeline import Pipeline, node
from .nodes import clean_X, clean_y


# since our data is split into train and test sets we'll need
# to clean them seperately
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_X,
                inputs="X_train",
                outputs="X_train_cleaned",
                name="X_train_cleaning_node",
            ),
            node(
                clean_X,
                inputs="X_test",
                outputs="X_test_cleaned",
                name="X_test_cleaning_node",
            ),
            node(
                clean_y,
                inputs="y_train",
                outputs="y_train_cleaned",
                name="y_train_cleaning_node",
            ),
            node(
                clean_y,
                inputs="y_test",
                outputs="y_test_cleaned",
                name="y_test_cleaning_node",
            ),
        ]
    )
