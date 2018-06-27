import numpy as np


class MetricError(Exception):
    pass


def lift_score(y_target, y_predicted):
    if not isinstance(y_target, np.ndarray) or isinstance(y_predicted, np.ndarray):
        raise MetricError("Input must be a numpy NDArray")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)
