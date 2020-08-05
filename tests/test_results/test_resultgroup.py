import pathlib
from typing import List

import pytest
from sklearn.linear_model import LogisticRegression

from ml_tooling.data import load_demo_dataset, Dataset
from ml_tooling.metrics import Metrics
from ml_tooling.result import Result, ResultGroup


class TestResultGroup:
    @pytest.fixture(scope="class")
    def dataset(self):
        """Setup a Dataset"""
        return load_demo_dataset("iris")

    @pytest.fixture(scope="class")
    def result_1(self, dataset: Dataset):
        """Setup a result"""
        return Result(
            estimator=LogisticRegression(),
            metrics=Metrics.from_dict({"accuracy": 0.5, "roc_auc": 0.5}),
            data=dataset,
        )

    @pytest.fixture(scope="class")
    def result_2(self, dataset: Dataset):
        """Setup a second Result"""
        return Result(
            estimator=LogisticRegression(),
            metrics=Metrics.from_dict({"accuracy": 0.8, "roc_auc": 0.1}),
            data=dataset,
        )

    @pytest.fixture(scope="class")
    def result_group(self, result_1: Result, result_2: Result):
        """Setup a ResultGroup"""
        return ResultGroup([result_1, result_2]).sort()

    @pytest.fixture(scope="class")
    def result_group_roc(self, result_1: Result, result_2: Result):
        return ResultGroup([result_1, result_2]).sort("roc_auc")

    @pytest.fixture()
    def logs(
        self, result_group: ResultGroup, tmp_path: pathlib.Path
    ) -> List[pathlib.Path]:
        """Setup a log"""
        result_group.log(tmp_path)
        return list(tmp_path.rglob("*.yaml"))

    def test_result_group_proxies_correctly_to_the_first_metric(
        self, result_group: ResultGroup
    ):
        """Expect that getting an attribute proxies to the first metric"""
        assert result_group.metrics.name == "accuracy"

    @pytest.mark.parametrize("index, metric_name", [(0, "accuracy"), (1, "roc_auc")])
    def test_result_group_can_index_into_metrics(
        self, index: int, metric_name: str, result_group: ResultGroup
    ):
        """Expect that metrics can be indexed into and get the correct metric name"""
        assert result_group.metrics[index].name == metric_name

    def test_result_group_logs_all_results(self, logs: List[pathlib.Path]):
        """Expect there will be two logs in the directory"""
        assert len(logs) == 2

    def test_result_group_logs_with_the_same_name(self, logs: List[pathlib.Path]):
        """Expect Result group to log with the same name"""
        assert all(["DemoData_LogisticRegression" in file.name for file in logs])

    @pytest.mark.parametrize("index, score", [(0, 0.8), (1, 0.5)])
    def test_result_group_sorts_by_first_metric(self, result_group, index, score):
        """Expect the highest accuracy to be the first result"""
        assert result_group.results[index].metrics.score == score

    def test_result_group_has_the_correct_order(
        self, result_group: ResultGroup, result_1: Result, result_2: Result
    ):
        assert result_group.results == [result_2, result_1]

    def test_result_group_can_be_sorted_by_other_metrics(
        self, result_group_roc: ResultGroup, result_1: Result, result_2: Result
    ):
        """Expect result group to be able to be sorted by a different metrics"""
        assert result_group_roc.results == [result_1, result_2]
