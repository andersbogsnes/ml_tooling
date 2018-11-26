from functools import total_ordering
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from .logging import log_model
from .plots import (_get_feature_importance,
                    plot_feature_importance,
                    plot_residuals,
                    plot_prediction_error,
                    plot_confusion_matrix,
                    VizError,
                    plot_roc_auc,
                    plot_lift_curve,
                    )
from .utils import _get_model_name, _get_labels


@total_ordering
class Result:
    """
    Represents a single scoring of a model.
    Contains plotting methods, as well as being comparable with other results
    """

    def __init__(self,
                 model,
                 score,
                 viz=None,
                 metric=None,
                 ):
        self.model = model
        self.model_name = _get_model_name(model)
        self.score = score
        self.metric = metric
        self.plot = viz

    @property
    def model_params(self) -> dict:
        """
        Calls get_params on estimator. Checks if estimator is a Pipeline, in which case it
        assumes last step in pipeline is an estimator and calls get_params on that step only
        :return:
            dict of params
        """
        if isinstance(self.model, Pipeline):
            return self.model.steps[-1][1].get_params()
        return self.model.get_params()

    def log_model(self, run_dir):
        metric_score = {self.metric: float(self.score)}
        return log_model(metric_scores=metric_score,
                         model_name=self.model_name,
                         model_params=self.model_params,
                         run_dir=run_dir,
                         )

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Output result as a dataframe for ease of inspecting and manipulating.
        Defaults to including model params, which can be toggled with the params flag.
        This is useful if you're comparing different models
        :param params:
            Whether or not to include model parameters as columns.
        :return:
            DataFrame of the result
        """
        model_params_dict = {}
        if params:
            model_params_dict = self.model_params

        model_params_dict['score'] = self.score
        model_params_dict['metric'] = self.metric

        return pd.DataFrame([model_params_dict])

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
                 ):
        self.cv = cv
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = np.mean(cross_val_scores)
        self.cross_val_std = np.std(cross_val_scores)
        super().__init__(model=model,
                         viz=viz,
                         score=self.cross_val_mean,
                         metric=metric,
                         )

    def to_dataframe(self, params: bool = True, cross_val_score: bool = False) -> pd.DataFrame:
        """
        Output result as a DataFrame for ease of inspecting and manipulating.
        Defaults to including model params, which can be toggled with the params flag.
        This is useful if you're comparing different models. Additionally includes
        the standard deviation of the score and number of cross validations.

        If you want to inspect the cross-validated scores, toggle cross_val_score and the resulting
        DataFrame will have one row per fold.
        :param params:
            Boolean - whether to include model parameters as columns in the DataFrame or not
        :param cross_val_score:
            Boolean - whether to have one row per fold in the DataFrame or not
        :return:
            DataFrame of results
        """
        df = super().to_dataframe(params).assign(cross_val_std=self.cross_val_std, cv=self.cv)
        if cross_val_score:
            return pd.concat([df.assign(score=score)
                              for score in self.cross_val_scores], ignore_index=True)
        return df

    def __repr__(self):
        cross_val_type = f"{self.cv}-fold " if isinstance(self.cv, int) else ''
        return f"<Result {self.model_name}: " \
               f"{cross_val_type}Cross-validated {self.metric}: {np.round(self.score, 2)} " \
               f"± {np.round(self.cross_val_std, 2)}>"


class ResultGroup:
    """
    A container for results. Proxies attributes to the best result. Supports indexing like a list.
    Can output the mean score of all its results using .mean_score.
    Can convert the results to a DataFrame of results, for ease of scanning and manipulating

    """

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

    def __getitem__(self, item):
        return self.results[item]

    def __repr__(self):
        results = '\n'.join([str(result) for result in self.results])
        return f"[{results}]"

    def log_model(self, log_dir):
        for result in self.results:
            result.log_model(log_dir)

    def mean_score(self):
        """
        Calculates mean score across the results
        :return:
        """
        return np.mean([result.score for result in self.results])

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Outputs results as a DataFrame. By default, the DataFrame will contain
        all possible model parameters. This behaviour can be toggled using `params=False`

        :param params:
            Boolean toggling whether or not to output params as part of the DataFrame
        :return:
            pd.DataFrame of results
        """

        output = [result.to_dataframe(params) for result in self.results]

        return pd.concat(output, ignore_index=True).sort_values(by='score', ascending=False)


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, model, config, data):
        self._model = model
        self._model_name = _get_model_name(model)
        self._config = config
        self._data = data

    def feature_importance(self,
                           values: bool = True,
                           top_n: Union[int, float] = None,
                           bottom_n: Union[int, float] = None,
                           n_samples=None,
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

        :param n_samples:


        :return:
            matplotlib.Axes
        """

        title = f"Feature Importance - {self._model_name}"
        importance = _get_feature_importance(self, n_samples)
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
