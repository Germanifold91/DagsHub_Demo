"""Data Training pipeline for DagsHubDemo"""

import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


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


def data_transform(
    training_data: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Transforms the training data by encoding categorical features and scaling numerical features,
    and imputing missing values. Returns the transformed data as a Pandas DataFrame and the fitted ColumnTransformer.

    Parameters:
    - training_data: pd.DataFrame - The input dataframe.
    - params: Dict[str, Any] - Parameters including 'categorical_cols' and 'numerical_cols' lists.

    Returns:
    - Tuple[pd.DataFrame, ColumnTransformer]: The transformed DataFrame and the fitted ColumnTransformer.
    """
    # Extract column lists from parameters
    categorical_cols = params["categorical_cols"]
    numerical_cols = params["numerical_cols"]

    # Define imputers for numerical and categorical columns
    numerical_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    # Define transformations for the numerical and categorical columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", categorical_imputer),  # First impute missing values
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # Then encode
        ]
    )
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", numerical_imputer),  # First impute missing values
            ("scaler", StandardScaler()),  # Then scale
        ]
    )

    # Combine transformers in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Fit the preprocessor to the training data
    preprocessor.fit(training_data)

    # Transform data
    df_transformed = preprocessor.transform(training_data)

    # Generate column names for the transformed DataFrame
    new_categorical_features = preprocessor.named_transformers_["categorical"][
        "encoder"
    ].get_feature_names_out(categorical_cols)
    new_columns = (
        numerical_cols + new_categorical_features.tolist()
    )  # Concatenate numerical and new categorical column names
    new_columns = [col.replace(" ", "_") for col in new_columns]
    # Convert the transformed data back to a DataFrame
    df_transformed = pd.DataFrame(
        df_transformed, columns=new_columns, index=training_data.index
    )

    return df_transformed, preprocessor
