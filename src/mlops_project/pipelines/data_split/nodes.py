from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def two_step_proportions(train_p, val_p, test_p):
    """
    Since we need to split the data in two steps
    this function returns the proportions of the
    'test_size' arg. needed to get the true prop.
    
    Args:
        train_p: proportion of the data to be used for training
        val_p: proportion of the data to be used for validation
        test_p: proportion of the data to be used for testing
    
    Returns:
        A tuple with the proportions needed for the
        train_test_split function in the two steps.
    """ 
    return (test_p, 1-(train_p/(1-test_p)))


def split_data(
    df: pd.DataFrame,
    target_label: str,
    set_sizes: Tuple[float, float, float],
    stratify: bool,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training, validation and test sets.

    Args:
        df: The dataset to be split.
        target_label: The name of the target column.
        set_sizes: A tuple containing the proportions of the data to be used
            for training, validation and testing, respectively.
        stratify: Whether to stratify the split according to the target column.
        random_state: The seed to be used for random number generation.

    Returns:
        A tuple containing the training, val and test datasets (X and y).
    """
    X = df.drop(columns=[target_label])
    y = df[target_label]
    
    test_size_1, test_size_2 = two_step_proportions(*set_sizes)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size_1,
        shuffle=True,
        stratify=y,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=test_size_2,
        shuffle=True,
        stratify=y_train_val,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
