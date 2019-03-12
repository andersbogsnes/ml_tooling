import numpy as np

from ml_tooling.utils import DataType
from .utils import MetricError


def lift_score(y_target: DataType, y_predicted: DataType) -> float:
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
    y_target = np.array(y_target)
    y_predicted = np.array(y_predicted)

    if y_target.ndim > 1 or y_predicted.ndim > 1:
        raise MetricError("Input must be 1-dimensional")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)