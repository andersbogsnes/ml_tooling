from typing import Any, List

import attr
from sklearn.base import is_classifier

from ml_tooling.data import Dataset
from ml_tooling.logging.log_estimator import Log
from ml_tooling.metrics import Metrics
from ml_tooling.plots.viz import ClassificationVisualize, RegressionVisualize
from ml_tooling.utils import Estimator, _get_estimator_name


@attr.s(repr=False)
class Result:
    """
    Contains the result of a given training run.
    Contains plotting methods, as well as being comparable with other results

    Parameters
    ----------

    estimator: Estimator
        Estimator used to generate the result

    metrics: Metrics
        Metrics used to score the model

    data: Dataset
        Dataset used to generate the result
    """

    estimator: Estimator = attr.ib(eq=False)
    metrics: Metrics = attr.ib()
    data: Dataset = attr.ib(eq=False)

    @property
    def plot(self):
        if is_classifier(self.estimator):
            return ClassificationVisualize(self.estimator, self.data)
        return RegressionVisualize(self.estimator, self.data)

    @property
    def model(self):
        from ml_tooling import Model

        model = Model(self.estimator)
        model.result = self
        return model

    @property
    def parameters(self):
        return self.estimator.get_params()

    @classmethod
    def from_estimator(
        cls,
        estimator: Estimator,
        data: Dataset,
        metrics: List[str],
        cv: Any = None,
        n_jobs: int = None,
        verbose: int = 0,
    ) -> "Result":
        metrics = Metrics.from_list(metrics)
        if cv:
            metrics.score_metrics_cv(
                estimator=estimator,
                x=data.train_x,
                y=data.train_y,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            metrics.score_metrics(estimator=estimator, x=data.test_x, y=data.test_y)

        return cls(metrics=metrics, estimator=estimator, data=data)

    def log(self, saved_estimator_path=None, savedir=None) -> Log:
        log = Log.from_result(result=self, estimator_path=saved_estimator_path)
        if savedir:
            log.save_log(savedir)
        return log

    def __repr__(self):
        metrics = {
            name: round(value, 2) for name, value in self.metrics.to_dict().items()
        }
        return f"<Result {_get_estimator_name(self.estimator)}: {metrics}>"
