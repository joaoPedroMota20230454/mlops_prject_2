from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame, target: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and test sets.

    Args:
        df: The dataset to be split.
        test_size: The proportion of the dataset to include in the test split.
        random_state: The seed used by the random number generator.

    Returns:
        A tuple containing the training and test datasets.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
