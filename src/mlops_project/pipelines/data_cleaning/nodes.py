'''
Functions to clean the data.
'''

import pandas as pd
import numpy as np


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Columns to drop straight away.
    
    Args:
        df: pd.DataFrame: Dataframe to drop columns from.
    
    Returns:
        pd.DataFrame: Dataframe with columns dropped.
    '''
    
    columns_to_drop = ['weight',
                       'payer_code',
                       'medical_specialty']

    df = df.drop(columns=columns_to_drop, axis=1)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the 'gender' column.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    
    gender_replace = {'Male':0,
                      'Female':1,
                      'Unknown/Invalid':1}
    
    df['gender'] = df['gender'].replace(gender_replace)
    return df


def encode_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Ordinal encoding of the 'age' column.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    dict_age = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    
    df['age'] = df['age'].replace(dict_age)
    return df


def drop_unknown_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Drops rows with unknown diagnosis.
    
    Args:
        df: pd.DataFrame: Dataframe to drop rows from.
    
    Returns:
        pd.DataFrame: Dataframe with rows dropped.
    '''
    
    df = df.loc[df['diag_1'] != '?', :]
    df = df.loc[df['diag_2'] != '?', :]
    df = df.loc[df['diag_3'] != '?', :]
    
    return df


def encode_race(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the 'race' column.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    
    # Also dropping unknown races
    df = df.loc[df['race'] != '?', :]
    
    dict_replace_race = {
        'Caucasian': 0,
        'AfricanAmerican': 1,
        'Other': 2,
        'Asian': 3,
        'Hispanic': 4
    }

    df['race'] = df['race'].replace(dict_replace_race)
    return df


def encode_medication_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the medication columns. Additioanlly, drops columns
    with only one unique value.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    
    med_transform_1 = {
        'No': 0,
        'Steady': 1,
        'Up': 1,
        'Down': 1
    }

    med_transform_2 = {
        'No': 0,
        'Steady': 1,
    }

    med_transform_3 = {
        'No': 0,
        'Steady': 1,
        'Up': 1,
    }
    
    # during notebook exploration, this cols only had 1
    # unique value, # so they dont add any information
    df = df.drop(columns=['examide', 'citoglipton'])

    # apply transform 1
    df['metformin'] = df['metformin'].replace(med_transform_1)
    df['repaglinide'] = df['repaglinide'].replace(med_transform_1)
    df['nateglinide'] = df['nateglinide'].replace(med_transform_1)
    df['chlorpropamide'] = df['chlorpropamide'].replace(med_transform_1)
    df['glimepiride'] = df['glimepiride'].replace(med_transform_1)
    df['glipizide'] = df['glipizide'].replace(med_transform_1)
    df['glyburide'] = df['glyburide'].replace(med_transform_1)
    df['tolbutamide'] = df['tolbutamide'].replace(med_transform_2)
    df['rosiglitazone'] = df['rosiglitazone'].replace(med_transform_1)
    df['acarbose'] = df['acarbose'].replace(med_transform_1)
    df['miglitol'] = df['miglitol'].replace(med_transform_1)
    df['insulin'] = df['insulin'].replace(med_transform_1)
    df['glyburide-metformin'] = df['glyburide-metformin'].replace(med_transform_1)
    df['pioglitazone'] = df['pioglitazone'].replace(med_transform_1)

    # apply transform 2
    df['acetohexamide'] = df['acetohexamide'].replace(med_transform_2)
    df['tolbutamide'] = df['tolbutamide'].replace(med_transform_2)
    df['troglitazone'] = df['troglitazone'].replace(med_transform_2)
    df['glipizide-metformin'] = df['glipizide-metformin'].replace(med_transform_2)
    df['glimepiride-pioglitazone'] = df['glimepiride-pioglitazone'].replace(med_transform_2)
    df['metformin-rosiglitazone'] = df['metformin-rosiglitazone'].replace(med_transform_2)
    df['metformin-pioglitazone'] = df['metformin-pioglitazone'].replace(med_transform_2)

    # apply transform 3
    df['tolazamide'] = df['tolazamide'].replace(med_transform_3)
    
    return df


def encode_diabetes_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the diabetes columns.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    dict_diabetes_med = {
        'No': 0,
        'Yes': 1
    }
    
    df['change'] = df['change'].replace(dict_diabetes_med)
    
    dict_change_transform = {
        'No': 0,
        'Ch': 1
    }

    df['change'] = df['change'].replace(dict_change_transform)
    return df


def encode_test_results(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the test results columns.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    dict_transform_a1cresult = {
        'Norm': 1,
        '>7': 2,
        '>8': 3,
        np.nan: 0
    }

    df['A1Cresult'] = df['A1Cresult'].replace(dict_transform_a1cresult)

    dict_max_glu_serum = {
        'Norm': 1,
        '>200': 2,
        '>300': 3,
        np.nan: 0
    }

    df['max_glu_serum'] = df['max_glu_serum'].replace(dict_max_glu_serum)
    return df


def fix_readmitted(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fixes the 'readmitted' column.
    
    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.
    
    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    '''
    dict_readmited_transform = {
        'NO': 0,
        '>30': 1,
        '<30': 1
    }

    df['readmitted'] = df['readmitted'].replace(dict_readmited_transform)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleans the data.
    
    Args:
        df: pd.DataFrame: Dataframe to clean.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    '''
    
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
    
    return df