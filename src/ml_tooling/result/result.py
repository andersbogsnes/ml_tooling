import attr

from ml_tooling.data import Dataset
from ml_tooling.logging.log_estimator import Log
from ml_tooling.metrics import Metrics
from ml_tooling.result.viz import create_plotter, BaseVisualize


@attr.s()
class Result:
    """
    Contains the result of a given training run.
    Contains plotting methods, as well as being comparable with other results
    """

    model = attr.ib()
    metrics: Metrics = attr.ib()
    data: Dataset = attr.ib()
    plot: BaseVisualize = attr.ib()

    @classmethod
    def from_model(
        cls, model, data: Dataset, metrics: Metrics, cv=None, n_jobs=None, verbose=None
    ):
        if cv:
            metrics.score_estimator_cv(
                estimator=model.estimator,
                x=data.train_x,
                y=data.train_y,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            metrics.score_estimator(
                estimator=model.estimator, x=data.test_x, y=data.test_y
            )

        return cls(
            metrics=metrics, model=model, plot=create_plotter(model, data), data=data
        )

    def log(self, saved_estimator_path=None):
        return Log.from_result(result=self, estimator_path=saved_estimator_path)
