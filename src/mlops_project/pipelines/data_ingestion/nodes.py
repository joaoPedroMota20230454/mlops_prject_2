import pandas as pd


def import_raw_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Imports the raw data.
    
    Args:
        file_path: str: Path to the raw data.
    
    Returns:
        pd.DataFrame: Raw data.
    """
    
    return pd.read_csv(file_path, **kwargs)
