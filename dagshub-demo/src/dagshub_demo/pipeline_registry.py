"""Project pipelines."""

from typing import Dict
from platform import python_version


from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory
from dagshub_demo import __version__ as PROJECT_VERSION


from .pipelines.data_engineering.pipeline import (
    data_processing as data_enginering_pipeline,
)
from .pipelines.data_modeling.pipeline import (
    ml_pipeline as model_training_pipeline,
)
from .pipelines.user_app.pipeline import create_user_app_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_enginering_pipeline()
    ml_pipeline = model_training_pipeline()
    data_transforming_pipeline = ml_pipeline.only_nodes_with_tags("data_transformation")
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")

    training_pipeline_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="sample_heart_disease@pd",
        log_model_kwargs=dict(
            artifact_path="dagshub_demo",
            conda_env={
                "python": python_version(),
                "build_dependencies": ["pip"],
                "dependencies": [f"dagshub_demo=={PROJECT_VERSION}"],
            },
            signature="auto",
        ),
    )
    user_app_pipeline = create_user_app_pipeline()
    return {
        "data_processing": data_processing_pipeline,
        "data_transforming": data_transforming_pipeline,
        "training": training_pipeline_ml,
        "inference": inference_pipeline,
        "user_app": user_app_pipeline,
        "__default__": data_processing_pipeline
        + data_transforming_pipeline
        + training_pipeline_ml
        + inference_pipeline
        + user_app_pipeline,
    }
