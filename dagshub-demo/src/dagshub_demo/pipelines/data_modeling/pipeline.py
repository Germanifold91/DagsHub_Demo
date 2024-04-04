"""Modeling pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .data_training import split_data


def model_training(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["heart_disease@pd", "params:data_split"],
                outputs=["x_train@pd", "x_test@pd", "y_train@pd", "y_test@pd"],
                name="data_split",
                tags=["training", "data_spliting"],
            ),
        ]
    )
