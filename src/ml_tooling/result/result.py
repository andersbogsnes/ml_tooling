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
    def from_model(cls, model, data, metrics):
        for metric in metrics:
            metric.score_metric(model.estimator, data.test_x, data.test_y)

        return cls(
            metrics=metrics, model=model, plot=create_plotter(model, data), data=data
        )

    def log(self, saved_estimator_path=None):
        return Log.from_result(result=self, estimator_path=saved_estimator_path)
