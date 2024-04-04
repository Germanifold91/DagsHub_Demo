"""Data Inference for DagsHubDemo"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer


def predict_model(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
    inference_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Makes predictions using a trained model and returns the predictions as a DataFrame.

    Parameters:
    - model: The trained model to use for prediction.
    """
    # Make predictions
    predictions = model.predict(preprocessor.transform(inference_data))

    # Return predictions as a DataFrame
    return pd.DataFrame({"predictions": predictions})
