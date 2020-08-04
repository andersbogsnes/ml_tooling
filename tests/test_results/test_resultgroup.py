import pathlib

from ml_tooling import Model
from ml_tooling.metrics import Metrics
from ml_tooling.result import Result, ResultGroup
from ml_tooling.utils import _get_estimator_name


class TestResultGroup:
    def test_result_group_proxies_correctly(
        self, train_iris_dataset, classifier: Model, classifier_cv: Model
    ):
        result1 = Result.from_estimator(
            classifier.estimator, train_iris_dataset, metrics=["accuracy"],
        )
        result2 = Result.from_estimator(
            classifier_cv.estimator, train_iris_dataset, metrics=["accuracy"],
        )

        group = ResultGroup([result1, result2])
        result_name = _get_estimator_name(group.estimator)
        assert result_name == "LogisticRegression"

    def test_result_group_implements_indexing_properly(
        self, train_iris_dataset, classifier: Model
    ):
        result1 = Result.from_estimator(
            estimator=classifier.estimator,
            data=train_iris_dataset,
            metrics=["accuracy"],
        )

        result2 = Result.from_estimator(
            estimator=classifier.estimator,
            data=train_iris_dataset,
            metrics=["accuracy"],
        )

        group = ResultGroup([result1, result2])

        assert group[0].metrics.score == result1.metrics.score
        assert group[1].metrics.score == result2.metrics.score

    def test_result_group_logs_all_results(
        self, tmp_path: pathlib.Path, train_iris_dataset, classifier: Model
    ):
        runs = tmp_path / "runs"
        result1 = Result.from_estimator(
            estimator=classifier.estimator,
            data=train_iris_dataset,
            metrics=["accuracy"],
        )
        result2 = Result.from_estimator(
            estimator=classifier.estimator,
            data=train_iris_dataset,
            metrics=["accuracy"],
        )

        group = ResultGroup([result1, result2])
        group.log(runs)

        run_files = list(runs.rglob("IrisData_LogisticRegression*"))

        assert len(run_files) == 2
        assert all(("IrisData_LogisticRegression" in file.name for file in run_files))

    def test_logging_a_result_works_as_expected(self, classifier: Model):
        result = classifier.result
        log = result.log()

        assert log.name == "IrisData_LogisticRegression"
        assert "accuracy" in log.metrics
        assert log.estimator_path is None

    def test_resultgroup_sorts_correctly(self, classifier):
        result1 = Result(
            classifier.estimator,
            metrics=Metrics.from_dict({"accuracy": 0.9, "roc_auc": 0.1}),
            data=None,
        )
        result2 = Result(
            classifier.estimator,
            metrics=Metrics.from_dict({"accuracy": 0.5, "roc_auc": 0.5}),
            data=None,
        )

        result3 = Result(
            classifier.estimator,
            metrics=Metrics.from_dict({"accuracy": 0.1, "roc_auc": 0.9}),
            data=None,
        )
        results = ResultGroup([result1, result2, result3])

        results.sort()
        assert results.results == [result1, result2, result3]

        results.sort(by="roc_auc")
        assert results.results == [result3, result2, result1]

    def test_can_instantiate_model_from_result(self, classifier: Model):
        result = Result(
            classifier.estimator,
            metrics=Metrics.from_dict({"accuracy": 0.1, "roc_auc": 0.9}),
            data=None,
        )

        assert isinstance(result.model, Model)
        assert result.model.estimator == classifier.estimator