from kedro.pipeline import Pipeline, node
from .nodes import clean_data, split_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_data,
                inputs="hospital_raw_data",
                outputs="hospital_cleaned_encoded_data",
                name="clean_data_node",
            ),
            node(
                split_data,
                inputs="hospital_cleaned_encoded_data",
                outputs=["hospital_train_data", "hospital_val_data", "hospital_test_data"],
                name="split_data_node",
            ),

        ]
    )