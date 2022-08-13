import os
import numpy as np
import pandas as pd
from typing import Tuple


def get_splits(
    data_dir: str, cancer: str, project: str, n_samples: int, split_number: int, setting: str = "standard"
) -> Tuple:
    """Return train and test indices to use for each "split" of the dataset in a cross-validation setting.

    Args:
        data_dir (str): Data directory path (root folder). Follow dataset download instructions and maintain the folder structure.
        cancer (str): Cancer for which the splits are needed.
        project (str): Project to which the cancer belongs.
        n_samples (int): The size of the dataset being split.
        split_number (int): The cross-validation split number for which indices are needed.
        setting (str, optional): The setting for which splits are needed. One of 'standard', 'missing' or 'pancancer'. Defaults to "standard".

    Returns:
        Tuple: Tuple containing the training set indices without missing data, with missing data (if setting is "pancancer" or "missing"), and the test indices.
    """
    assert project in ["TCGA", "ICGC", "TARGET"], NotImplementedError("Project not found.")
    assert setting in ["standard", "missing", "pancancer"], NotImplementedError("Invalid setting.")

    train_splits = pd.read_csv(os.path.join(data_dir, f"splits/{project}/{cancer}_train_splits.csv"))
    test_splits = pd.read_csv(os.path.join(data_dir, f"splits/{project}/{cancer}_test_splits.csv"))

    train_ix = train_splits.iloc[split_number, :].dropna().values.astype(int)
    test_ix = test_splits.iloc[split_number, :].dropna().values.astype(int)

    if setting in ["missing", "pancancer"]:
        missing_start_idx = max(list(train_ix) + list(test_ix))
        assert missing_start_idx >= n_samples, ValueError("Please ensure the missing samples are correcly appended.")

        train_missing_combined_ix = np.append(train_ix, np.array(range(missing_start_idx + 1, n_samples)))

        return train_ix, train_missing_combined_ix, test_ix

    return train_ix, test_ix
