import pytest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from ml_tooling.data import Dataset
from ml_tooling.search import GridSearch, RandomSearch
from scipy.stats.distributions import randint
from itertools import product


class TestGridSearch:
    """Tests covering the GridSearch searcher"""

    @pytest.fixture()
    def grid(self, pipeline_forest_classifier: Pipeline) -> GridSearch:
        """Setup a GridSearch object"""
        return GridSearch(
            pipeline_forest_classifier, param_grid={"estimator__max_depth": [1, 2, 3]}
        )

    def test_prepare_gridsearch_estimators_has_different_parameters(self, grid):
        """Expect that each estimator has a different max_depth"""
        estimators = grid.prepare_gridsearch_estimators()

        for estimator, penalty in zip(estimators, [1, 2, 3]):
            assert estimator.get_params()["estimator__max_depth"] == penalty

    def test_grid_can_take_a_cv_object(
        self, grid: GridSearch, train_iris_dataset: Dataset
    ):
        """Expect that grid will use a given CV splitter if passed"""
        cv = KFold(n_splits=2)
        results = grid.search(
            train_iris_dataset, metrics=["accuracy"], cv=cv, n_jobs=-1
        )
        assert len(results[0].metrics[0].cross_val_scores) == 2

    def test_grid_returns_correct_number_of_results(
        self, grid, train_iris_dataset: Dataset
    ):
        """Expect that grid will return 3 results, one for each parameter"""
        results = grid.search(train_iris_dataset, metrics=["accuracy"], cv=2, n_jobs=-1)
        assert len(results) == 3


class TestRandomSearch:
    """Tests covering the RandomSearch searcher"""

    @pytest.fixture()
    def grid(self, pipeline_forest_classifier: Pipeline):
        """Setup a RandomSearch searcher"""
        return RandomSearch(
            pipeline_forest_classifier,
            param_grid={
                "estimator__max_depth": randint(0, 100),
                "estimator__criterion": ["gini", "entropy"],
            },
            n_iter=2,
        )

    def test_random_search_has_different_hyperparams(
        self, grid: RandomSearch, train_iris_dataset: Dataset
    ):
        """
        Expect that the randomsearch estimators will have different hyperparameters
        each time the estimators are built
        """
        estimators = [grid.prepare_randomsearch_estimators() for _ in range(5)]
        for a, b in zip(estimators, estimators[1:]):
            for estimator_a, estimator_b in product(a, b):
                assert estimator_a.get_params() != estimator_b.get_params()

    def test_random_search_can_take_a_cv_object(
        self, grid: RandomSearch, train_iris_dataset: Dataset
    ):
        """Expect that RandomSearch will use a given CV splitter if passed"""
        cv = KFold(n_splits=2)
        results = grid.search(
            train_iris_dataset, metrics=["accuracy"], cv=cv, n_jobs=-1
        )
        assert len(results[0].metrics[0].cross_val_scores) == 2

    def test_random_search_returns_correct_number_of_results(
        self, grid: RandomSearch, train_iris_dataset: Dataset
    ):
        """Expect that RandomSearch will return 2 results, equal to n_iter"""
        results = grid.search(train_iris_dataset, metrics=["accuracy"], cv=2, n_jobs=-1)
        assert len(results) == grid.n_iter
