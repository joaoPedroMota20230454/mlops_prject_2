"""
This is a boilerplate test file for pipeline 'data_unit_tests_after_cleaning'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
from kedro.framework.context import KedroContext
from src.mlops_project.pipelines.data_cleaning.nodes import encode_gender, encode_age_bracket, map_diagnosis_to_bin, drop_unwanted_columns, encode_race, encode_medication_columns, encode_diabetes_columns, encode_test_results, fix_readmitted, clean_df
import os
import pandas as pd
import numpy as np





def test_encode_gender():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_gender(df_sample)
    assert set(result.gender.unique().tolist()) == {0, 1}

def test_encode_age_bracket():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_age_bracket(df_sample)
    assert set(result.age.unique().tolist()) == {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}




def test_map_diagnosis_to_bin():
    # Test cases for starts with 'E' or 'V'
    assert map_diagnosis_to_bin('E123') == 49
    assert map_diagnosis_to_bin('V123') == 49

    # Test case for NO_DIAGNOSIS
    assert map_diagnosis_to_bin('NO_DIAGNOSIS') == 0

    # Test cases for ranges
    assert map_diagnosis_to_bin('50.0') == 1
    assert map_diagnosis_to_bin('140.1') == 2
    assert map_diagnosis_to_bin('250') == 24

    # Test cases for diabetes mellitus (250.x)
    assert map_diagnosis_to_bin('250.0') == 25
    assert map_diagnosis_to_bin('250.1') == 26
    assert map_diagnosis_to_bin('250.20') == 8
    assert map_diagnosis_to_bin('250.21') == 8
    assert map_diagnosis_to_bin('250.22') == 9

    # Test cases for other ranges
    assert map_diagnosis_to_bin('251.0') == 3
    assert map_diagnosis_to_bin('285.9') == 35
    assert map_diagnosis_to_bin('300.0') == 36
    assert map_diagnosis_to_bin('350.0') == 37
    assert map_diagnosis_to_bin('400.0') == 38
    assert map_diagnosis_to_bin('490.0') == 39
    assert map_diagnosis_to_bin('530.0') == 40
    assert map_diagnosis_to_bin('590.0') == 41
    assert map_diagnosis_to_bin('650.0') == 42
    assert map_diagnosis_to_bin('690.0') == 43
    assert map_diagnosis_to_bin('720.0') == 44
    assert map_diagnosis_to_bin('750.0') == 45
    assert map_diagnosis_to_bin('770.0') == 46
    assert map_diagnosis_to_bin('790.0') == 47
    assert map_diagnosis_to_bin('850.0') == 48



def test_drop_unwanted_columns():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = drop_unwanted_columns(df_sample)
    assert {"weight",
        "payer_code",
        "medical_specialty",
        "patient_nbr"} not in set(result.columns.tolist())


def test_encode_race():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_race(df_sample)

    assert set(result.race.unique().tolist()) == {0, 1, 2, 3, 4}

def test_encode_medication_columns():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_medication_columns(df_sample)
    # test transform 1
    assert set(result["metformin"].unique().tolist()) == {0, 1}
    assert set(result["repaglinide"].unique().tolist()) == {0, 1}
    assert set(result["nateglinide"].unique().tolist()) == {0, 1}
    assert set(result["chlorpropamide"].unique().tolist()) == {0, 1}
    assert set(result["glimepiride"].unique().tolist()) == {0, 1}
    assert set(result["glyburide"].unique().tolist()) == {0, 1}
    assert set(result["glipizide"].unique().tolist()) == {0, 1}
    assert set(result["tolbutamide"].unique().tolist()) == {0, 1}
    assert set(result["rosiglitazone"].unique().tolist()) == {0, 1}
    assert set(result["acarbose"].unique().tolist()) == {0, 1}
    assert set(result["miglitol"].unique().tolist()) == {0, 1}
    assert set(result["insulin"].unique().tolist()) == {0, 1}
    assert set(result["glyburide-metformin"].unique().tolist()) == {0, 1}
    assert set(result["pioglitazone"].unique().tolist()) == {0, 1}

    # test transform 2
    #assert set(result["acetohexamide"].unique().tolist()) == {0, 1}
    assert set(result["tolbutamide"].unique().tolist()) == {0, 1}
    #assert set(result["troglitazone"].unique().tolist()) == {0, 1}
    assert set(result["glipizide-metformin"].unique().tolist()) == {0, 1}
    #assert set(result["glimepiride-pioglitazone"].unique().tolist()) == {0, 1}
    #assert set(result["metformin-rosiglitazone"].unique().tolist()) == {0, 1}
    #assert set(result["metformin-pioglitazone"].unique().tolist()) == {0, 1}

    # test transform 3
    assert set(result["tolazamide"].unique().tolist()) == {0, 1}


def test_encode_diabetes_columns():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_diabetes_columns(df_sample)
    assert set(result["diabetesMed"].unique().tolist()) == {0, 1}
    assert set(result["change"].unique().tolist()) == {0, 1}

def test_encode_test_results():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_test_results(df_sample)
    assert set(result["A1Cresult"].unique().tolist()) == {0, 1, 2, 3}
    assert set(result["max_glu_serum"].unique().tolist()) == {0, 1, 2, 3}



def test_fix_readmitted():
    filepath = os.path.join("tests/sample/target_sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = fix_readmitted(df_sample)
    assert set(result["readmitted"].unique().tolist()) == {0, 1}

def test_clean_df():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    filepath_target_data = os.path.join("tests/sample/target_sample_raw_data.csv")
    df_target = pd.read_csv(filepath_target_data)
    data, target = clean_df(df_sample, df_target)
    
    # check that all columns are of type int/float/bool
    
    cols = data.columns.tolist()
    for col in cols:
        assert data[col].dtype in [np.int64, np.int32, np.float64, np.float32, np.bool_]
    assert target.dtype in [np.int64, np.int32, np.float64, np.float32, np.bool_]


# if __name__ == '__main__':
#     test_encode_gender()