import numpy as np
from sklearn import metrics


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, normalized=True
) -> np.ndarray:
    """
    Generates a confusion matrix using sklearn's confusion_matrix function, with the ability
    to add normalization.

    Calculates a matrix of
        [[True Positives, False Positives],
         [False Negatives, True Negatives]]

    :param y_true:
        True labels

    :param y_pred:
        Predicted labels
    :param normalized:
        Whether or not to normalize counts

    :return:
        Confusion matrix array
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalized is True:
        cm = cm / cm.sum()
        cm = np.around(cm, 2)
        cm[np.isnan(cm)] = 0.0
    return cm
