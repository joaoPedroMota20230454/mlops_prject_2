import pytest
import pandas as pd
import numpy as np
import os
from src.mlops_project.pipelines.feature_engineering.nodes import total_visits_in_previous_year, add_features


def test_total_visits_in_previous_year():
    def contains_negative_values(df, col):
        return df[col].lt(0).any()
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)
    df_sample = total_visits_in_previous_year(df_sample)

    assert not contains_negative_values(df_sample, 'total_visits_in_previous_year'), "A soma está a obter resultados negativos. Este número não pode ser negativo."




    
def test_add_features():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)
    

    df_sample_modified = add_features(df_sample, upload_to_feature_store=False)
    
    len_df_sample = len(df_sample.columns)

    len_df_sample_modified = len(df_sample_modified.columns)
    assert len_df_sample_modified > len_df_sample, "O número de colunas não está a aumentar. Verifique se está a adicionar as features corretamente."



if __name__ == "__main__":
    test_add_features