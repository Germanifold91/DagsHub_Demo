"""Modeling pipeline."""

from kedro.pipeline import Pipeline, node
from .data_training import (
    split_data,
    data_transform,
    train_model,
    evaluate_model,
)
from .inference import predict_model


def ml_pipeline(**kwargs) -> Pipeline:
    pipeline_transformation = Pipeline(
        [
            node(
                func=split_data,
                inputs=["training_heart_disease@pd", "params:data_split"],
                outputs=["x_train@pd", "x_test@pd", "y_train@pd", "y_test@pd"],
                name="data_split",
                tags=["data_transforming"],
            ),
            node(
                func=data_transform,
                inputs=["x_train@pd", "x_test@pd", "params:data_transform"],
                outputs=[
                    "training_transformed_df@pd",
                    "test_transformed_df@pd",
                    "transformer",
                ],
                name="data_transformer",
                tags=["data_transforming", "training"],
            ),
        ]
    )
    pipeline_training = Pipeline(
        [
            node(
                func=train_model,
                inputs=["training_transformed_df@pd", "y_train@pd", "params:modeling"],
                outputs="best_model",
                name="model_training",
                tags=["training", "modeling"],
            ),
            node(
                func=evaluate_model,
                inputs=["best_model", "test_transformed_df@pd", "y_test@pd"],
                outputs=[
                    "precision_weighted",
                    "recall_weighted",
                    "accuracy",
                    "f1_score_weighted",
                    "confusion_matrix",
                ],
                name="model_evaluation",
                tags=["training", "evaluation"],
            ),
        ]
    )
    inference_pipeline = Pipeline(
        [
            node(
                func=predict_model,
                inputs=["best_model", "transformer", "sample_heart_disease@pd"],
                outputs="model_inference",
                name="model_inference",
                tags=["inference"],
            ),
        ]
    )

    return pipeline_transformation + pipeline_training + inference_pipeline
