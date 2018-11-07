from functools import total_ordering
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from .plots import (_get_feature_importance,
                    plot_feature_importance,
                    plot_residuals,
                    plot_prediction_error,
                    plot_confusion_matrix,
                    VizError,
                    plot_roc_auc,
                    plot_lift_curve,
                    )
from .utils import get_model_name, _get_labels


@total_ordering
class Result:
    def __init__(self,
                 model,
                 score,
                 viz=None,
                 metric=None,
                 labels=None
                 ):
        self.model = model
        self.model_name = get_model_name(model)
        self.score = score
        self.metric = metric
        self.plot = viz
        self.labels = labels

    @property
    def model_params(self):
        if isinstance(self.model, Pipeline):
            return self.model.steps[-1][1].get_params()
        return self.model.get_params()

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"<Result {self.model_name}: " \
               f"{self.metric}: {np.round(self.score, 2)} >"


class CVResult(Result):
    """
    Data class for holding results of model testing.
    Also implements comparison operators for finding max mean score
    """

    def __init__(self,
                 model,
                 viz=None,
                 cv=None,
                 cross_val_scores=None,
                 metric=None,
                 labels=None
                 ):
        self.cv = cv
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = np.mean(cross_val_scores)
        self.cross_val_std = np.std(cross_val_scores)
        super().__init__(model=model,
                         viz=viz,
                         score=self.cross_val_mean,
                         metric=metric,
                         labels=labels)

    def __repr__(self):
        cross_val_type = f"{self.cv}-fold " if isinstance(self.cv, int) else ''
        return f"<Result {self.model_name}: " \
               f"{cross_val_type}Cross-validated {self.metric}: {np.round(self.score, 2)} " \
               f"± {np.round(self.cross_val_std, 2)}>"


class ResultGroup:
    def __init__(self, results: List[Result]):
        self.results = sorted(results, reverse=True)

    def __getattr__(self, name):
        return getattr(self.results[0], name)

    def __dir__(self):
        proxied_dir = dir(self.results[0])
        custom_methods = ['to_dataframe', 'mean_score']
        return proxied_dir + custom_methods

    def __len__(self):
        return len(self.results)

    def mean_score(self):
        return np.mean([result.score for result in self.results])

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Outputs results as a DataFrame. By default, the dataframe will contain
        all possible model parameters. This behaviour can be toggled using `params=False`

        :param params:
            Boolean controlling whether or not to output params as part of the dataframe
        :return:
            pd.DataFrame of results
        """
        def create_row(result, param):
            data_dict = {"model": result.model_name,
                         "score": result.score}
            if param:
                data_dict.update(result.model_params)

            if isinstance(result, CVResult):
                data_dict["cross_val_std"] = result.cross_val_std
                data_dict['cv'] = result.cv

            return data_dict

        output = [create_row(result, params) for result in self.results]

        return pd.DataFrame(output).sort_values(by='score', ascending=False)


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, model, config, data):
        self._model = model
        self._model_name = get_model_name(model)
        self._config = config
        self._data = data

    def feature_importance(self,
                           values: bool = True,
                           top_n: Union[int, float] = None,
                           bottom_n: Union[int, float] = None,
                           **kwargs) -> plt.Axes:
        """
        Visualizes feature importance of the model. Model must have either feature_importance_
        or coef_ attribute

        :param values:
            Toggles value labels on end of each bar

        :param top_n:
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        :param bottom_n:
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        :return:
            matplotlib.Axes
        """

        title = f"Feature Importance - {self._model_name}"
        importance = _get_feature_importance(self._model)
        labels = _get_labels(self._model, self._data.train_x)

        with plt.style.context(self._config.STYLE_SHEET):
            return plot_feature_importance(importance,
                                           labels,
                                           values=values,
                                           title=title,
                                           top_n=top_n,
                                           bottom_n=bottom_n,
                                           **kwargs)


class RegressionVisualize(BaseVisualize):
    """
    Visualization class for Regression models
    """

    def residuals(self, **kwargs) -> plt.Axes:
        """
        Visualizes residuals of a regression model.
        Any kwargs are passed onto matplotlib

        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Residual Plot - {self._model_name}"
            y_pred = self._model.predict(self._data.test_x)
            return plot_residuals(self._data.test_y, y_pred, title, **kwargs)

    def prediction_error(self, **kwargs) -> plt.Axes:
        """
        Visualizes prediction error of a regression model
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Prediction Error - {self._model_name}"
            y_pred = self._model.predict(self._data.test_x)
            return plot_prediction_error(self._data.test_y, y_pred, title=title, **kwargs)


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """

    def confusion_matrix(self, normalized: bool = True, **kwargs) -> plt.Axes:
        """
        Visualize a confusion matrix for a classification model
        Any kwargs are passed onto matplotlib
        :param normalized:
            Whether or not to normalize annotated class counts

        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f'Confusion Matrix - {self._model_name}'
            y_pred = self._model.predict(self._data.test_x)
            return plot_confusion_matrix(self._data.test_y, y_pred, normalized, title, **kwargs)

    def roc_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a ROC curve for a classification model.
        Model must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        if not hasattr(self._model, 'predict_proba'):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config.STYLE_SHEET):
            title = f'ROC AUC - {self._model_name}'
            y_proba = self._model.predict_proba(self._data.test_x)[:, 1]
            return plot_roc_auc(self._data.test_y, y_proba, title=title, **kwargs)

    def lift_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Lift Curve for a classification model
        Model must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f'Lift Curve - {self._model_name}'
            y_proba = self._model.predict_proba(self._data.test_x)[:, 1]
            return plot_lift_curve(self._data.test_y, y_proba, title=title, **kwargs)
