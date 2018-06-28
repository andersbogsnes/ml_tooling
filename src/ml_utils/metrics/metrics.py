import numpy as np
from sklearn import metrics


class MetricError(Exception):
    pass


def lift_score(y_target, y_predicted):
    if not isinstance(y_target, np.ndarray) or not isinstance(y_predicted, np.ndarray):
        raise MetricError("Input must be a numpy NDArray")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)


def confusion_matrix(y_true, y_pred, normalized=True):
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalized is True:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        cm[np.isnan(cm)] = 0.0
    return cm


def sorted_feature_importance(labels, importance):
    idx = np.argsort(np.abs(importance))
    return labels[idx], importance[idx]
