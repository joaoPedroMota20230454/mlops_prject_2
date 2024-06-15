import os
from pathlib import Path
from typing import Union

import pandas as pd

import hopsworks

import great_expectations as gx
from great_expectations.core import ExpectationSuite

from kedro.config import OmegaConfigLoader, MissingConfigException
from kedro.framework.project import settings


def read_credentials(path: Union[str, Path] = None) -> dict:
    """
    Read credentials from a YAML file.

    Args:
        path (Union[str, Path], optional): Path to the YAML file. Defaults to None.
        key (str, optional): Key to read from the credentials. Defaults to None.
    
    Returns:
        dict: Dictionary with the credentials.
    """
    
    # code sourced from: https://docs.kedro.org/en/stable/configuration/credentials.html
    
    if path is None:
        conf_path = str(Path(os.getcwd()) / settings.CONF_SOURCE)
    else:
        conf_path = str(Path(path) / settings.CONF_SOURCE)
        
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    
    try:
        credentials = conf_loader["credentials"]
    except MissingConfigException:
        credentials = {}
    
    return credentials


def load_expectation_suite(suite_name: str) -> ExpectationSuite:
    """
    Load an expectation suite from the Great Expectations context.
    
    Args:
        suite_name (str): Name of the expectation suite.
    
    Returns:
        ExpectationSuite: Expectation suite.
    """
    context = gx.get_context()
    suite = context.get_expectation_suite(expectation_suite_name=suite_name)
    return suite


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    SETTINGS: dict
) -> None:
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    
    
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.
    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    # TODO: do we need to return anything?
    # return object_feature_group


if __name__ == '__main__':
    print(read_credentials())
    