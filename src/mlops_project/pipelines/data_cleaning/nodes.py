from typing import Tuple
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Columns to drop straight away.

    Args:
        df: pd.DataFrame: Dataframe to drop columns from.

    Returns:
        pd.DataFrame: Dataframe with columns dropped.
    """

    columns_to_drop = ["weight",
        "payer_code",
        "medical_specialty",
        "patient_nbr"]

    df = df.drop(columns=columns_to_drop, axis=1)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Encodes the 'gender' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    gender_replace = {"Male": 0, "Female": 1, "Unknown/Invalid": 1}

    df["gender"] = df["gender"].replace(gender_replace)
    return df


def encode_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Ordinal encoding of the 'age' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_age = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    df["age"] = df["age"].replace(dict_age)
    return df


def drop_unknown_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with unknown diagnosis.

    Args:
        df: pd.DataFrame: Dataframe to drop rows from.

    Returns:
        pd.DataFrame: Dataframe with rows dropped.
    """

    df = df[df["diag_1"] != "?"]
    df = df[df["diag_2"] != "?"]
    df = df[df["diag_3"] != "?"]

    return df


def map_diagnosis_to_bin(diagnosis):
    """
    TESTED? YES
    Map diagnosis to bins

    ### Args:
        - `diagnosis (str)`: Diagnosis code.

    ### Returns:
        - `int`: Diagnosis bin.
    """
    if diagnosis.startswith(('E', 'V')):
        return 49
    if "NO_DIAGNOSIS" in diagnosis:
        return 0
    diagnosis_full = diagnosis.split(".")
    diagnosis = int(diagnosis_full[0])
    if 1 <= diagnosis <= 139:
        return 1
    elif 140 <= diagnosis <= 239:
        return 2
    elif 240 <= diagnosis <= 249:
        return 3
    elif diagnosis == 250:
        list_of_numbers = []
        if len(diagnosis_full) == 2:
            for letter in diagnosis_full[1]:
                list_of_numbers.append(letter)
        if len(list_of_numbers) == 0:
            return 24
        elif len(list_of_numbers) == 1:
            if list_of_numbers[0] == "0":
                return 25
            elif list_of_numbers[0] == "1":
                return 26
            elif list_of_numbers[0] == "2":
                return 27
            elif list_of_numbers[0] == "3":
                return 28
            elif list_of_numbers[0] == "4":
                return 29
            elif list_of_numbers[0] == "5":
                return 30
            elif list_of_numbers[0] == "6":
                return 31
            elif list_of_numbers[0] == "7":
                return 32
            elif list_of_numbers[0] == "8":
                return 33
            elif list_of_numbers[0] == "9":
                return 34
        else:
            if list_of_numbers[0] == "0":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 4
                else:
                    return 5
            elif list_of_numbers[0] == "1":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 6
                else:
                    return 7
            elif list_of_numbers[0] == "2":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 8
                else:
                    return 9
            elif list_of_numbers[0] == "3":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 10
                else:
                    return 11
            elif list_of_numbers[0] == "4":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 12
                else:
                    return 13
            elif list_of_numbers[0] == "5":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 14
                else:
                    return 15
            elif list_of_numbers[0] == "6":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 16
                else:
                    return 17
            elif list_of_numbers[0] == "7":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 18
                else:
                    return 19
            elif list_of_numbers[0] == "8":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 20
                else:
                    return 21
            elif list_of_numbers[0] == "9":
                if list_of_numbers[1] == "0" or list_of_numbers[1] == "1":
                    return 22
                else:
                    return 23
    elif 251 <= diagnosis <= 279:
        return 3
    elif 280 <= diagnosis <= 289:
        return 35
    elif 290 <= diagnosis <= 319:
        return 36
    elif 320 <= diagnosis <= 389:
        return 37
    elif 390 <= diagnosis <= 459:
        return 38
    elif 460 <= diagnosis <= 519:
        return 39
    elif 520 <= diagnosis <= 579:
        return 40
    elif 580 <= diagnosis <= 629:
        return 41
    elif 630 <= diagnosis <= 679:
        return 42
    elif 680 <= diagnosis <= 709:
        return 43
    elif 710 <= diagnosis <= 739:
        return 44
    elif 740 <= diagnosis <= 759:
        return 45
    elif 760 <= diagnosis <= 779:
        return 46
    elif 780 <= diagnosis <= 799:
        return 47
    elif 800 <= diagnosis <= 999:
        return 48




def encode_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Encodes the 'race' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    # Also dropping unknown races
    df = df.loc[df["race"] != "?", :]

    dict_replace_race = {
        "Caucasian": 0,
        "AfricanAmerican": 1,
        "Other": 2,
        "Asian": 3,
        "Hispanic": 4,
    }

    df["race"] = df["race"].replace(dict_replace_race)
    return df


def encode_medication_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Encodes the medication columns. Additioanlly, drops columns
    with only one unique value.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    med_transform_1 = {"No": 0, "Steady": 1, "Up": 1, "Down": 1}

    med_transform_2 = {
        "No": 0,
        "Steady": 1,
    }

    med_transform_3 = {
        "No": 0,
        "Steady": 1,
        "Up": 1,
    }

    # during notebook exploration, this cols only had 1
    # unique value, # so they dont add any information
    df = df.drop(columns=["examide", "citoglipton"])

    # apply transform 1
    df["metformin"] = df["metformin"].replace(med_transform_1)
    df["repaglinide"] = df["repaglinide"].replace(med_transform_1)
    df["nateglinide"] = df["nateglinide"].replace(med_transform_1)
    df["chlorpropamide"] = df["chlorpropamide"].replace(med_transform_1)
    df["glimepiride"] = df["glimepiride"].replace(med_transform_1)
    df["glipizide"] = df["glipizide"].replace(med_transform_1)
    df["glyburide"] = df["glyburide"].replace(med_transform_1)
    df["tolbutamide"] = df["tolbutamide"].replace(med_transform_2)
    df["rosiglitazone"] = df["rosiglitazone"].replace(med_transform_1)
    df["acarbose"] = df["acarbose"].replace(med_transform_1)
    df["miglitol"] = df["miglitol"].replace(med_transform_1)
    df["insulin"] = df["insulin"].replace(med_transform_1)
    df["glyburide-metformin"] = df["glyburide-metformin"].replace(med_transform_1)
    df["pioglitazone"] = df["pioglitazone"].replace(med_transform_1)

    # apply transform 2
    df["acetohexamide"] = df["acetohexamide"].replace(med_transform_2)
    df["tolbutamide"] = df["tolbutamide"].replace(med_transform_2)
    df["troglitazone"] = df["troglitazone"].replace(med_transform_2)
    df["glipizide-metformin"] = df["glipizide-metformin"].replace(med_transform_2)
    df["glimepiride-pioglitazone"] = df["glimepiride-pioglitazone"].replace(
        med_transform_2
    )
    df["metformin-rosiglitazone"] = df["metformin-rosiglitazone"].replace(
        med_transform_2
    )
    df["metformin-pioglitazone"] = df["metformin-pioglitazone"].replace(med_transform_2)

    # apply transform 3
    df["tolazamide"] = df["tolazamide"].replace(med_transform_3)

    return df


def encode_diabetes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Encodes the diabetes columns.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_diabetes_med = {"No": 0, "Yes": 1}

    df["diabetesMed"] = df["diabetesMed"].replace(dict_diabetes_med)

    dict_change_transform = {"No": 0, "Ch": 1}

    df["change"] = df["change"].replace(dict_change_transform)
    return df


def encode_test_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    TESTED? YES
    Encodes the test results columns.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_transform_a1cresult = {"Norm": 1, ">7": 2, ">8": 3, np.nan: 0}

    df["A1Cresult"] = df["A1Cresult"].replace(dict_transform_a1cresult)

    dict_max_glu_serum = {"Norm": 1, ">200": 2, ">300": 3, np.nan: 0}

    df["max_glu_serum"] = df["max_glu_serum"].replace(dict_max_glu_serum)
    return df


def fix_readmitted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the 'readmitted' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_readmited_transform = {"NO": 0, ">30": 1, "<30": 1}

    df["readmitted"] = df["readmitted"].replace(dict_readmited_transform)
    return df


def clean_df(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the data.

    Args:
        X: pd.DataFrame: Features.
        y: pd.DataFrame: Target.
    
    Returns:
    """
    df = X.merge(y, left_index=True, right_index=True)
    cleaning_functions = [
        drop_unwanted_columns,
        encode_gender,
        encode_age_bracket,
        drop_unknown_diagnosis,
        encode_race,
        encode_medication_columns,
        encode_diabetes_columns,
        encode_test_results,
        fix_readmitted
    ]

    for func in cleaning_functions:
        df = func(df)

    # map diagnosis to bins
    df["diag_1"] = df["diag_1"].apply(map_diagnosis_to_bin)
    df["diag_2"] = df["diag_2"].apply(map_diagnosis_to_bin)
    df["diag_3"] = df["diag_3"].apply(map_diagnosis_to_bin)
    
    X  = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    return X, y
