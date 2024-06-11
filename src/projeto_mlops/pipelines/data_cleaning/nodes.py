import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.drop(columns=['weight', 'payer_code', 'medical_specialty',], inplace=True)
    data.gender.replace({'Male':0, "Female":1, "Unknown/Invalid":1}, inplace=True)

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
    data.age.replace(dict_age, inplace=True)
    # diagnosis
    data = data[data["diag_1"] != "?"]
    data = data[data["diag_2"] != "?"]
    data = data[data["diag_3"] != "?"]
    # race
    data = data[data["race"] != "?"]


    med_transform_1 = {
        "No": 0,
        "Steady": 1,
        "Up": 1,
        "Down": 1
    }

    med_transform_2 = {
        "No": 0,
        "Steady": 1,
    }

    med_transform_3 = {
        "No": 0,
        "Steady": 1,
        "Up": 1,
    }
    # during notebook exploration, this cols only had 1 unique value, so they dont add any information
    data.drop(columns=["examide", "citoglipton",], inplace=True)

    data["metformin"].replace(med_transform_1, inplace=True)
    data["repaglinide"].replace(med_transform_1, inplace=True)
    data["nateglinide"].replace(med_transform_1, inplace=True)
    data["chlorpropamide"].replace(med_transform_1, inplace=True)
    data["glimepiride"].replace(med_transform_1, inplace=True)
    data["glipizide"].replace(med_transform_1, inplace=True)
    data["glyburide"].replace(med_transform_1, inplace=True)
    data["tolbutamide"].replace(med_transform_2, inplace=True)
    data["rosiglitazone"].replace(med_transform_1, inplace=True)
    data["acarbose"].replace(med_transform_1, inplace=True)
    data["miglitol"].replace(med_transform_1, inplace=True)
    data["insulin"].replace(med_transform_1, inplace=True)
    data["glyburide-metformin"].replace(med_transform_1, inplace=True)


    data['acetohexamide'].replace(med_transform_2, inplace=True)
    data['tolbutamide'].replace(med_transform_2, inplace=True)
    data['troglitazone'].replace(med_transform_2, inplace=True)
    data["glipizide-metformin"].replace(med_transform_2, inplace=True)
    data["glimepiride-pioglitazone"].replace(med_transform_2, inplace=True)
    data["metformin-rosiglitazone"].replace(med_transform_2, inplace=True)
    data["metformin-pioglitazone"].replace(med_transform_2, inplace=True)

    data['tolazamide'].replace(med_transform_3, inplace=True)


    # ['Caucasian', 'AfricanAmerican', '?', 'Other', 'Asian', 'Hispanic']
    dict_replace_race = {
        "Caucasian": 0,
        "AfricanAmerican": 1,
        "Other": 2,
        "Asian": 3,
        "Hispanic": 4
    }

    dict_change_transform = {"No": 0, "Ch": 1}
    dict_diabetes_med = {"No": 0, "Yes": 1}
    dict_readmited_transform = {"NO": 0, ">30": 1, "<30": 1}

    data["change"] = data["change"].replace(dict_change_transform)
    data["diabetesMed"] = data["diabetesMed"].replace(dict_diabetes_med)
    data["readmitted"] = data["readmitted"].replace(dict_readmited_transform)

    dict_transform_a1cresult = {"Norm": 1, ">7": 2, ">8": 3, np.nan: 0}

    data["A1Cresult"] = data["A1Cresult"].replace(dict_transform_a1cresult)

    dict_max_glu_serum = {"Norm": 1, ">200": 2, ">300": 3, np.nan: 0}

    data["max_glu_serum"] = data["max_glu_serum"].replace(dict_max_glu_serum)


    return data



def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    X = data.drop(columns=["readmitted"])
    y = data["readmitted"]
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    # concatenating the data back together
    X_train = pd.concat([X_train, y_train], axis=1)
    X_val = pd.concat([X_val, y_val], axis=1)
    X_test = pd.concat([X_test, y_test], axis=1)

    return X_train, X_val, X_test
