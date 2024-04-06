"""User App pipeline for DagsHubDemo."""

from kedro.pipeline import Pipeline, node
from .model_consumption import predict_with_mlflow


def create_user_app_pipeline(**kwargs):
    pipeline_user_app = Pipeline(
        [
            node(
                func=predict_with_mlflow,
                inputs=dict(
                    model="pipeline_inference_model", data="sample_heart_disease@pd"
                ),
                outputs="predictions_mlflow",
                name="predict_with_mlflow",
                tags="user_app",
            )
        ]
    )

    return pipeline_user_app
