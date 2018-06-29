import numpy as np
from sklearn import metrics


class MetricError(Exception):
    pass


def lift_score(y_target: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates lift score for a given model. The lift score quantifies how much better
    the model is compared to a random baseline.

    The formula is defined as follows:
        lift = (TP/(TP+FN)(TP+FP)/(TP+TN+FP+FN)
        Source: https://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score

    :param y_target:
        Target labels

    :param y_predicted:
        Predicted labels

    :return:
        Lift score
    """
    if not isinstance(y_target, np.ndarray) or not isinstance(y_predicted, np.ndarray):
        raise MetricError("Input must be a numpy NDArray")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalized=True) -> np.ndarray:
    """
    Generates a confusion matrix using sklearns confusion_matrix function, with the ability
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
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        cm[np.isnan(cm)] = 0.0
    return cm


def sorted_feature_importance(labels, importance):
    idx = np.argsort(np.abs(importance))
    return labels[idx], importance[idx]
