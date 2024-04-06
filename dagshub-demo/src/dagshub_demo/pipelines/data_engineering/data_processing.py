"""Data Processing for DagsHub Demo"""

import pandas as pd
from typing import Tuple


# Sample dataset
def sample_df(
    data_frame: pd.DataFrame, sample_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample a given number of rows from a DataFrame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame to sample from.
    sample_size : int
        The number of rows to sample. If this is greater than 20% of the
        number of rows in the DataFrame, it will be capped at that amount.

    Returns
    -------
    pd.DataFrame
        The sampled DataFrame.
    pd.DataFrame
        The original DataFrame with the sampled rows removed.
    """

    data_frame = data_frame.drop(["id", "dataset"], axis=1)
    print(f"Input DataFrame:\n{data_frame}")
    print(f"Sample size: {sample_size}")

    if sample_size > int(len(data_frame) * 0.2):
        print(
            f"Sample size was greater than 20% of the DataFrame,\
                capping at {int(len(data_frame) * 0.2)}"
        )
        sample_size = int(len(data_frame) * 0.2)

    print("Sampling data...")
    sampled_df = data_frame.sample(sample_size)
    sampled_df = sampled_df.drop("num", axis=1)

    print("Dropping sampled rows...")
    original_df = data_frame.drop(sampled_df.index)

    return sampled_df, original_df
