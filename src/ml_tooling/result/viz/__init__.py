from .baseviz import BaseVisualize
from .regression_viz import RegressionVisualize
from .classification_viz import ClassificationVisualize


def create_plotter(model, data):
    if model.is_classifier:
        return ClassificationVisualize(model.estimator, data, model.config)
    return RegressionVisualize(model.estimator, data, model.config)


__all__ = [
    "BaseVisualize",
    "RegressionVisualize",
    "ClassificationVisualize",
    "create_plotter",
]
