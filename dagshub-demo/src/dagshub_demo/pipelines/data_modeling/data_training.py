"""Data Training pipeline for DagsHubDemo"""

import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to split the data into training and test sets
def split_data(
    data_frame: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the given data frame into training and test sets.

    Args:
        data_frame (pd.DataFrame): The input data frame.
        params (Dict[str, Any]): The parameters containing target column, test size, and random state.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The split data frames and series.
    """
    # Extracting parameters
    target_column = params["target_column"]  # Target column name
    test_size = params["test_size"]  # Test size
    random_state = params["random_state"]  # Random state

    # Splitting features and target
    X = data_frame.drop(target_column, axis=1)  # Features
    y = data_frame[target_column]  # Target

    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
