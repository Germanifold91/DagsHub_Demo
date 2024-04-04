"""Data Processing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .data_processing import sample_df


def data_processing(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_df,
                inputs=["heart_disease@pd", "params:sample_size"],
                outputs=["sample_heart_disease@pd", "training_heart_disease@pd"],
                name="data_sampling",
                tags=["data_engineering", "data_sampling"],
            ),
        ]
    )
