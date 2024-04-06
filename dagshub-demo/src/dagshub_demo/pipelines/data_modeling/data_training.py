"""Data Training pipeline for DagsHubDemo"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Function to split the data into training and test sets
def split_data(
    data_frame: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the given data frame into training and test sets.

    Args:
        data_frame (pd.DataFrame): The input data frame.
        params (Dict[str, Any]): The parameters containing target column,
        test size, and random state.

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
    training_data: pd.DataFrame, test_data: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Transforms the training and test data by encoding categorical features, scaling numerical features,
    and imputing missing values using a specified strategy. Supports KNN imputation for numerical columns.

    Parameters:
    - training_data: pd.DataFrame - The input training dataframe.
    - test_data: pd.DataFrame - The input test dataframe.
    - params: Dict[str, Any] - Parameters including 'categorical_cols', 'numerical_cols', and 'impute_strategy_num'.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]: The transformed training and test DataFrames and the fitted ColumnTransformer.
    """
    categorical_cols = params["categorical_cols"]
    numerical_cols = params["numerical_cols"]
    impute_strategy_num = params["impute_strategy_num"]

    # Determine the imputer for numerical columns based on the specified strategy
    if impute_strategy_num == "knn":
        numerical_imputer = KNNImputer()
    else:
        numerical_imputer = SimpleImputer(strategy=impute_strategy_num)

    categorical_imputer = SimpleImputer(strategy="most_frequent")

    categorical_transformer = Pipeline(
        [
            ("imputer", categorical_imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        [
            ("imputer", numerical_imputer),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_transformer, numerical_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    preprocessor.fit(training_data)

    train_transformed = preprocessor.transform(training_data)
    test_transformed = preprocessor.transform(test_data)

    new_categorical_features = preprocessor.named_transformers_["categorical"][
        "encoder"
    ].get_feature_names_out(categorical_cols)
    new_columns = numerical_cols + new_categorical_features.tolist()
    new_columns = [col.replace(" ", "_") for col in new_columns]

    train_transformed_df = pd.DataFrame(
        train_transformed, columns=new_columns, index=training_data.index
    )
    test_transformed_df = pd.DataFrame(
        test_transformed, columns=new_columns, index=test_data.index
    )

    return train_transformed_df, test_transformed_df, preprocessor


def configure_model(model_type: str, random_state: int) -> Tuple[Any, Dict[str, list]]:
    """
    Configures a machine learning model and its hyperparameter search space.

    Parameters:
    - model_type: The type of model to configure ('random_forest',
                 'logistic_regression', 'k_neighbors').
    - random_state: A seed for random number generation to ensure reproducibility.

    Returns:
    - A tuple containing the configured model and its hyperparameter space.
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            random_state=random_state, multi_class="multinomial", max_iter=1000
        )
        param_distributions = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        }
    elif model_type == "k_neighbors":
        model = KNeighborsClassifier()
        param_distributions = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, param_distributions


def train_model(
    X_train_transformed: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
) -> Any:
    """
    Trains a specified model on pre-split training data with hyperparameter tuning using Randomized Search.

    Parameters:
    - X_train_transformed: Training features DataFrame.
    - y_train: Training target Series.
    - params: Parameters including 'model_type', 'random_state', 'num_iter', 'num_folds', and 'scoring'.

    Returns:
    - The best model from RandomizedSearchCV.
    """
    model_type = params["model_type"]
    random_state = params.get("random_state", 42)
    num_iter = params["num_iter"]
    num_folds = params["num_folds"]
    scoring_metric = params["scoring"]

    # Configure the model and its hyperparameter space
    model, param_distributions = configure_model(model_type, random_state)

    # Set up Randomized Search
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=num_iter,
        cv=num_folds,
        scoring=scoring_metric,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    # Perform the search
    random_search.fit(X_train_transformed, y_train)

    # Best model
    best_model = random_search.best_estimator_

    print(f"Best Hyperparameters for {model_type}: {random_search.best_params_}")

    return best_model


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list
) -> plt.Figure:
    """
    Generates and plots a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig


def evaluate_model(model, X_test_transformed: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates a trained model and plots the confusion matrix with numerical classes represented as Class_1, Class_2, etc.

    Parameters:
    - model: The trained model to evaluate.
    - X_transformed_test: DataFrame containing test features.
    - y_test: Series containing true labels for the test set.

    Returns:
    - Dictionary containing precision, recall, accuracy, F1 score, and the confusion matrix plot figure.
    """
    # Generate predictions
    y_pred = model.predict(X_test_transformed)

    # Calculate metrics
    metrics = {
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Dynamically generate class labels based on unique values in y_test and y_pred
    unique_classes = np.unique(np.concatenate((np.unique(y_test), np.unique(y_pred))))
    class_labels = [f"Class_{int(cls)}" for cls in unique_classes]

    # Plot the confusion matrix
    fig = plot_confusion_matrix(y_test, y_pred, class_labels)
    metrics["confusion_matrix_fig"] = fig

    # Print metrics for convenience
    for metric, value in metrics.items():
        if metric != "confusion_matrix_fig":
            print(f"{metric.capitalize()}: {value:.4f}")

    return (
        metrics["precision"],
        metrics["recall"],
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["confusion_matrix_fig"],
    )
