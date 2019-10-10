from typing import Union, List, Dict, Optional, Any

import attr
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from ml_tooling.utils import Estimator, DataType


@attr.s
class Metric:
    """
    Represents a single metric, containing a metric name and its corresponding score.
    Can be instantiated using any sklearn-compatible `score_` strings

    A Metric knows how to generate it's own score by calling :meth:`score_metric`, passing
    an estimator, an X and a Y. A Metric can also get a cross-validated score by calling
    :meth:`score_metric_cv` and passing a CV value - either a `CV_` object or an int specifying
    number of folds

    Examples
    --------

    .. doctest::

        >>> from ml_tooling.metrics import Metric
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> metric = Metric('r2')
        >>> x = np.array([[1],[2],[3],[4]])
        >>> y = np.array([[2], [4], [6], [8]])
        >>> estimator = LinearRegression().fit(x, y)
        >>> metric.score_metric(estimator, x, y)
        Metric(metric='r2', score=1.0, cross_val_scores=None)
        >>> metric.score
        1.0
        >>> metric.metric
        'r2'

    .. doctest::

        >>> metric.score_metric_cv(estimator, x, y, cv=2)
        Metric(metric='r2', score=1.0, cross_val_scores=array([1., 1.]))
        >>> metric.score
        1.0
        >>> metric.metric
        'r2'
        >>> metric.cross_val_scores
        array([1., 1.])
        >>> metric.std
        0.0

    .. _CV: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    .. _score: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values # noqa
    """

    metric: str = attr.ib()
    score: float = attr.ib(default=None)
    cross_val_scores: Optional[np.ndarray] = attr.ib(default=None)

    def score_metric(self, estimator: Estimator, x: DataType, y: DataType) -> "Metric":
        """
        Calculates the score for this metric. Takes a fitted estimator, x and y values.
        Scores are calculated with sklearn metrics - using the string defined in `self.metric` to
        look up the appropriate scoring function.

        Parameters
        ----------
        estimator: Pipeline or BaseEstimator
            A fitted estimator to score
        x: np.ndarray, pd.DataFrame
            Features to score model with
        y: np.ndarray, pd.Series
            Target to score model with

        Returns
        -------
        self
        """
        scoring_func = get_scorer(self.metric)
        self.score = scoring_func(estimator, x, y)

        # Ensure metric is reset if called multiple times
        self.cross_val_scores = None
        return self

    @property
    def std(self):
        if self.cross_val_scores is not None:
            return np.std(self.cross_val_scores)

    def score_metric_cv(
        self,
        estimator: Estimator,
        x: DataType,
        y: DataType,
        cv: Any,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "Metric":
        """
        Score metric using cross-validation. When scoring with cross_validation,
        `self.cross_val_scores` is populated with the cross validated scores and `self.score` is
        set to the mean value of `self.cross_val_scores`. Cross validation can be parallelized by
        passing the `n_jobs` parameter

        Parameters
        ----------
        estimator: Pipeline or BaseEstimator
            Fitted estimator to score
        x: np.ndarray or pd.DataFrame
            Features to use in scoring
        y: np.ndarray or pd.Series
            Target to use in scoring
        cv: int, BaseCrossValidator
            If an int is passed, cross-validate using K-Fold with `cv` folds.
            If BaseCrossValidator is passed, use that object instead
        n_jobs: int
            Number of jobs to use in parallelizing. Pass None to not do CV in parallel
        verbose: int
            Verbosity level of output

        Returns
        -------
        self
        """
        self.cross_val_scores = cross_val_score(
            estimator, x, y, cv=cv, scoring=self.metric, n_jobs=n_jobs, verbose=verbose
        )
        self.score = float(np.mean(self.cross_val_scores))
        return self


@attr.s(auto_attribs=True)
class Metrics:
    """
    Represents a collection of :class:`Metric`. This is the default object used when scoring an
    estimator.

    There are two alternate constructors:
    - :meth:`from_list` takes a list of metric names and instantiates one metric per list item
    - :meth:`from_dict` takes a dictionary of name -> score and instantiates one metric with
    the given score per dictionary item

    Calling either :meth:`score_metrics` or :meth:`score_metrics_cv` will in turn call
    :meth:`~ml_tooling.metrics.metric.Metric.score_metric` or
    :meth:`~ml_tooling.metrics.metric.Metric.score_metric_cv` of each :class:`Metric` in its
    collection

    Examples
    --------
    To score multiple metrics, create a metrics object from a list and call :meth:`score_metrics`
    to score all metrics in one operation

    .. code-block::

        >>> from ml_tooling.metrics import Metrics
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> metrics = Metrics.from_list(['r2', 'neg_mean_squared_error'])
        >>> x = np.array([[1],[2],[3],[4]])
        >>> y = np.array([[2], [4], [6], [8]])
        >>> estimator = LinearRegression().fit(x, y)
        >>> metrics.score_metrics(estimator=estimator, x=x, y=y)
        >>> for metric in metrics:
        ...     print(metric)
        Metric(metric='r2', score=1.0, cross_val_scores=None)
        Metric(metric='neg_mean_squared_error', score=-0.0, cross_val_scores=None)

    We can convert metrics to a dictionary

    .. code-block::

        >>> metrics.to_dict()
        {'r2': 1.0, 'neg_mean_squared_error': -0.0}

    or a list

    .. code-block::

        >>> metrics.to_list()
        ['r2', 'neg_mean_squared_error']


    """

    metrics: List[Metric]

    @classmethod
    def from_list(cls, metrics: List[str]):
        return cls([Metric(metric=metric) for metric in metrics])

    @classmethod
    def from_dict(cls, metrics: Dict[str, Union[float, int]]):
        return cls([Metric(metric=key, score=value) for key, value in metrics.items()])

    def to_list(self):
        return [m.metric for m in self.metrics]

    def to_dict(self):
        return {
            m.metric: float(m.score) if m.score is not None else None
            for m in self.metrics
        }

    def score_metrics(self, estimator, x, y):
        for metric in self.metrics:
            metric.score_metric(estimator, x, y)

    def score_metrics_cv(self, estimator, x, y, cv, n_jobs=-1, verbose=0):
        for metric in self.metrics:
            metric.score_metric_cv(estimator, x, y, cv, n_jobs, verbose)

    def __len__(self):
        return len(self.metrics)

    def __getattr__(self, name):
        if hasattr(self.metrics[0], name):
            return getattr(self.metrics[0], name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def __getitem__(self, item):
        return self.metrics[item]

    def __contains__(self, item):
        for metric in self.metrics:
            if metric.metric == item:
                return True
        return False
