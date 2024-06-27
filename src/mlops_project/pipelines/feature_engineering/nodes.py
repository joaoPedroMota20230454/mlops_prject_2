import pandas as pd
from .utils import to_feature_store, read_credentials, load_expectation_suite


FEATURE_DESCRIPTIONS = [
    {"name": "race", "description": "Race of the patient"},
    {"name": "gender", "description": "Gender of the patient"},
    {"name": "age", "description": "Age of the patient"},
    {"name": "admission_type_id", "description": "Type of admission"},
    {"name": "discharge_disposition_id", "description": "Disposition after discharge"},
    {"name": "admission_source_id", "description": "Source of admission"},
    {"name": "time_in_hospital", "description": "Duration of the hospital stay in days"},
    {"name": "num_lab_procedures", "description": "Number of lab procedures performed during the hospital stay"},
    {"name": "num_procedures", "description": "Number of procedures performed during the hospital stay"},
    {"name": "num_medications", "description": "Number of medications prescribed during the hospital stay"},
    {"name": "number_outpatient", "description": "Number of outpatient visits in the previous year"},
    {"name": "number_emergency", "description": "Number of emergency visits in the previous year"},
    {"name": "number_inpatient", "description": "Number of inpatient visits in the previous year"},
    {"name": "diag_1", "description": "Primary diagnosis"},
    {"name": "diag_2", "description": "Secondary diagnosis"},
    {"name": "diag_3", "description": "Additional diagnosis"},
    {"name": "number_diagnoses", "description": "Number of diagnoses during the hospital stay"},
    {"name": "max_glu_serum", "description": "Maximum glucose serum test result"},
    {"name": "A1Cresult", "description": "A1C test result"},
    {"name": "metformin", "description": "Use of metformin medication"},
    {"name": "repaglinide", "description": "Use of repaglinide medication"},
    {"name": "nateglinide", "description": "Use of nateglinide medication"},
    {"name": "chlorpropamide", "description": "Use of chlorpropamide medication"},
    {"name": "glimepiride", "description": "Use of glimepiride medication"},
    {"name": "acetohexamide", "description": "Use of acetohexamide medication"},
    {"name": "glipizide", "description": "Use of glipizide medication"},
    {"name": "glyburide", "description": "Use of glyburide medication"},
    {"name": "tolbutamide", "description": "Use of tolbutamide medication"},
    {"name": "pioglitazone", "description": "Use of pioglitazone medication"},
    {"name": "rosiglitazone", "description": "Use of rosiglitazone medication"},
    {"name": "acarbose", "description": "Use of acarbose medication"},
    {"name": "miglitol", "description": "Use of miglitol medication"},
    {"name": "troglitazone", "description": "Use of troglitazone medication"},
    {"name": "tolazamide", "description": "Use of tolazamide medication"},
    {"name": "insulin", "description": "Use of insulin medication"},
    {"name": "glyburide-metformin", "description": "Use of glyburide-metformin combination medication"},
    {"name": "glipizide-metformin", "description": "Use of glipizide-metformin combination medication"},
    {"name": "glimepiride-pioglitazone", "description": "Use of glimepiride-pioglitazone combination medication"},
    {"name": "metformin-rosiglitazone", "description": "Use of metformin-rosiglitazone combination medication"},
    {"name": "metformin-pioglitazone", "description": "Use of metformin-pioglitazone combination medication"},
    {"name": "change", "description": "Indicates if there was a change in medication"},
    {"name": "diabetesMed", "description": "Indicates if diabetes medication was prescribed"},
    {"name": "total_visits_in_previous_year", "description": "Total number of outpatient, emergency, and inpatient visits in the previous year"}
]



def total_visits_in_previous_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the total number of visits in the previous year.
    
    Args:
        df: pd.DataFrame: Dataframe to calculate the feature for.
    
    Returns:
        pd.DataFrame: Dataframe with the new feature.
    """
    df['total_visits_in_previous_year'] = df['number_outpatient'] \
                                        + df['number_emergency'] \
                                        + df['number_inpatient']
    return df


def add_features(df: pd.DataFrame, upload_to_feature_store: bool) -> pd.DataFrame:
    """
    Adds new features to the dataframe.

    Args:
        df: pd.DataFrame: Dataframe to clean.

    Returns:
        pd.DataFrame: Engineered dataframe.
    """


    df_copy = df.copy()

    fe_functions = [
        total_visits_in_previous_year,
    ]
    
    # print("total_visits_in_previous_year" in df.columns)
    for func in fe_functions:
        df_copy = func(df_copy)
    # print("total_visits_in_previous_year" in df.columns)
    if upload_to_feature_store:
        # TODO should we add them as params??
        SETTINGS_STORE = read_credentials()["SETTINGS_STORE"]
        suite = load_expectation_suite("clean_suite")
        to_feature_store(
            data=df_copy,
            group_name="diabetes",
            feature_group_version=1,
            description="Diabetes dataset with additional features",
            group_description=FEATURE_DESCRIPTIONS,
            validation_expectation_suite=suite,
            SETTINGS=SETTINGS_STORE
        )

    return df_copy