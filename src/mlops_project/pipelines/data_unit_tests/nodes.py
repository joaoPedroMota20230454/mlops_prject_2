import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration
from great_expectations.core.batch import RuntimeBatchRequest

def test_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # 1. great_expectations init
    # 2. great_expectations datasource new
    # 3. great_expectations suite new
    # 4. great_expectations suite edit <suite name>
    
    # code sourced from: https://docs.greatexpectations.io/docs/oss/tutorials/quickstart/
    # also: https://github.com/datarootsio/tutorial-great-expectations/blob/main/tutorial_great_expectations.ipynb
    # also: https://medium.com/@mostsignificant/python-data-validation-made-easy-with-the-great-expectations-package-8d1be266fd3f
    
    # context = gx.get_context()

    # batch_request = RuntimeBatchRequest(
    #     datasource_name="raw_datasource",  # check the name by: CLI -> great_expectations datasource list
    #     data_connector_name="default_runtime_data_connector_name",
    #     data_asset_name="hospital",  # can be arbitrary name
    #     runtime_parameters={"batch_data": df},
    #     batch_identifiers={"default_identifier_name": "default_identifier"}
    # )
    
    # # Create the validator to validate the batch
    # validator = context.get_validator(
    #     batch_request=batch_request,
    #     expectation_suite_name="raw_suite"  # check the name by: CLI -> great_expectations suite list
    # )
    
    # skipping for now
    return df


if __name__ == "__main__":
    import os
    print(os.getcwd())
    df = pd.read_csv("data/01_raw/diabetic_data.csv")
    batch = test_data(df, "data/01_raw/diabetic_data.csv")
    print(batch.head())

    
    
