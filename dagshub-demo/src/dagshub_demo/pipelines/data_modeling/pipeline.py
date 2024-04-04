"""Modeling pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .data_training import split_data, data_transform


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
            node(
                func=data_transform,
                inputs=["x_train@pd", "params:data_transform"],
                outputs=["transformed_df@pd", "transformer"],
                name="data_transformer",
                tags=["training", "data_transforming"],
            ),
        ]
    )
