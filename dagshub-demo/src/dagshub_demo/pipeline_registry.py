"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.data_engineering import data_processing as data_egineering_pipeline
from .pipelines.data_modeling import model_training as ml_training_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_egineering_pipeline()
    ml_pipeline = ml_training_pipeline()
    return {
        "data_processing": data_processing_pipeline,
        "training": ml_pipeline,
        "__default__": data_processing_pipeline + ml_pipeline,
    }
