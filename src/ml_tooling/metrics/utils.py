import math
from typing import Union, Tuple, List

import numpy as np


def _get_top_n_idx(idx: np.ndarray, top_n: Union[int, float]) -> np.ndarray:
    """
    Gets top n idx, where n can be a float representing a percentage,
    or an int representing number of elements to return

    Parameters
    ----------
    idx: np.array
        Array of values to be indexed
    top_n: int, float
        Number of items to return. Floats are interpreted as percentage len(idx) to return


    Returns
    -------
    np.array
        Indexed array
    """

    if _is_percent(top_n):
        top_n: int = math.floor(
            top_n * len(idx)
        ) or 1  # If floor rounds to 0, use 1 instead
    return idx[:top_n]


def _get_bottom_n_idx(idx: np.array, bottom_n: Union[int, float]) -> np.array:
    """
    Gets top n idx, where n can be a float representing a percentage,
    or an int representing number of elements to return

    Parameters
    ----------
    idx: np.array
        Array of values to be indexed
    bottom_n: int, float
        Number of items to return. Floats are interpreted as percentage len(idx) to return


    Returns
    -------
    np.array
        Indexed array
    """

    if _is_percent(bottom_n):
        bottom_n = (
            math.floor(bottom_n * len(idx)) or 1
        )  # If floor rounds to 0, use 1 instead
    start_value = len(idx) - bottom_n
    return idx[start_value:]


def _sort_values(
    labels: np.ndarray,
    values: np.ndarray,
    abs_sort: bool = False,
    top_n: Union[int, float] = None,
    bottom_n: Union[int, float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts labels and values by values. Optionally specify whether to sort by absolute values,
    as well as getting top and/or bottom n values

    Parameters
    ----------

    labels: np.ndarray
        Array of feature labels

    values: np.ndarray
        Array of importance values

    abs_sort: str
        How data is sorted. Specify 'abs' to sort data by absolute value, else by numeric value

    top_n: int, float, optional
        If top_n is an int return top n features
        If top_n is a float between 0 and 1 return top top_n percent of features

    bottom_n: int, float, optional
        If bottom_n is an int return bottom n features
        If bottom_n is a float between 0 and 1 return bottom bottom_n percent of features

    Returns
    -------
    tuple(labels, values)
        List of labels and list of feature importances sorted by importance
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if abs_sort:
        idx = np.argsort(np.abs(values))[::-1]
    else:
        idx = np.argsort(values)[::-1]

    sorted_idx: List[int] = []

    if top_n:
        sorted_idx.extend(_get_top_n_idx(idx, top_n))

    if bottom_n:
        sorted_idx.extend(_get_bottom_n_idx(idx, bottom_n))

    if sorted_idx:
        idx = sorted_idx

    return labels[idx], values[idx]


def _is_percent(number: Union[float, int]) -> bool:
    """
    Checks if a value is a valid percent
    :param number:
        The number to validate
    :return:
        bool
    """
    if isinstance(number, float):
        if number > 1 or number < 0:
            raise ValueError(f"Floats only valid between 0 and 1. Got {number}")
        return True
    return False


