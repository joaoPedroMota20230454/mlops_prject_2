import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import BatchRequest

def test_data(df: pd.DataFrame,
              datasource_name: str,
              suite_name: str,
              data_asset_name: str,
              build_data_docs: bool) -> pd.DataFrame:
    """
    This function is used to test the data using great_expectations.
    
    Args:
        df: A pandas DataFrame containing the data.
        build_data_docs: A boolean indicating whether to build the data docs.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    
    Raises:
        ValueError: If the data validation fails.
    """
    
    # 1. great_expectations init
    # 2. great_expectations datasource new -> to add datasource, i.e. data/01_raw
    # 3. great_expectations suite new -> add expectations
    # 4. great_expectations suite edit <suite name> -> edit expectations
    
    # code sourced from: https://docs.greatexpectations.io/docs/oss/tutorials/quickstart/
    # also: https://github.com/datarootsio/tutorial-great-expectations/blob/main/tutorial_great_expectations.ipynb
    # also: https://medium.com/@mostsignificant/python-data-validation-made-easy-with-the-great-expectations-package-8d1be266fd3f
    
    context = gx.get_context()
    
    batch_request = {
        'datasource_name': datasource_name,  # The datasource name defined in the great_expectations.yml
        'data_connector_name': "default_inferred_data_connector_name",  # Also in the yaml
        'data_asset_name': f"{data_asset_name}.csv",  # The file name or asset name defined in the pattern
        'batch_spec_passthrough': {"reader_method": "read_csv"}  # Specify the reader method
    }
    
    # Create the validator to validate the batch
    validator = context.get_validator(
        batch_request=BatchRequest(**batch_request),
        expectation_suite_name=suite_name  # check the name by: CLI -> great_expectations suite list
    )
    
    # Validate the batch
    validation_result = validator.validate()

    # Build and open Data Docs to view validation results
    if build_data_docs:
        context.build_data_docs()
        context.open_data_docs()
    
    if not validation_result["success"]:
        raise ValueError("Data validation failed")
    
    return df


if __name__ == "__main__":
    # testing the function
    import os
    print(os.getcwd())
    df = pd.read_csv("data/01_raw/diabetic_data.csv")
    batch = test_data(df)
    print(batch.head())


    
    
