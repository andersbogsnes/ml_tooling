from typing import Union

import attr

from ml_tooling.data import Dataset
from ml_tooling.logging.log_estimator import Log
from ml_tooling.metrics import Metrics
from ml_tooling.result.viz import (
    create_plotter,
    ClassificationVisualize,
    RegressionVisualize,
)


@attr.s(repr=False)
class Result:
    """
    Contains the result of a given training run.
    Contains plotting methods, as well as being comparable with other results
    """

    model = attr.ib(eq=False)
    metrics: Metrics = attr.ib()
    data: Dataset = attr.ib(eq=False)
    plot: Union[ClassificationVisualize, RegressionVisualize] = attr.ib(
        eq=False, repr=False
    )

    @classmethod
    def from_model(
        cls, model, data: Dataset, metrics: Metrics, cv=None, n_jobs=None, verbose=None
    ) -> "Result":
        if cv:
            metrics.score_metrics_cv(
                estimator=model.estimator,
                x=data.train_x,
                y=data.train_y,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            metrics.score_metrics(
                estimator=model.estimator, x=data.test_x, y=data.test_y
            )

        return cls(
            metrics=metrics, model=model, plot=create_plotter(model, data), data=data
        )

    def log(self, saved_estimator_path=None, savedir=None) -> Log:
        log = Log.from_result(result=self, estimator_path=saved_estimator_path)
        if savedir:
            log.save_log(savedir)
        return log

    def __repr__(self):
        metrics = {
            name: round(value, 2) for name, value in self.metrics.to_dict().items()
        }
        return f"<Result {self.model.estimator_name}: {metrics}>"
