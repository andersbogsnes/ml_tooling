import numpy as np
import pandas as pd

from ml_tooling.utils import DataType


def target_correlation(
    features: pd.DataFrame, target: DataType, method: str = "pearson"
) -> pd.Series:
    """
    Calculate target_correlation between features and target and returns a sorted pd.Series

    Parameters
    ----------
    features: pd.DataFrame
        Features to calculate target_correlation for
    target: np.ndarray or pd.Series
        Target variable
    method: str
        Which correlation to use. One of 'pearson', 'spearman', 'kendall'

    Returns
    -------
    pd.Series
        Series of feature importance sorted by absolute value


    """
    if isinstance(target, np.ndarray):
        target = pd.Series(target)

    corr = features.corrwith(target, method=method)
    sorted_idx = np.argsort(corr.abs())
    return corr[sorted_idx]
