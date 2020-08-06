from typing import List, Union

import pytest
import sklearn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset, load_demo_dataset
from ml_tooling.plots.permutation_importance import plot_permutation_importance
from ml_tooling.result import Result
from ml_tooling.transformers import DFStandardScaler

feature_importances = ["0.56", "0.15", "0.01", "0.00"]


class TestPermutationImportancePlot:
    @pytest.fixture(scope="class")
    def dataset(self):
        return load_demo_dataset("iris")

    @pytest.fixture(scope="class")
    def classifier_result(self, dataset) -> Result:
        """Setup a classifier Result"""
        model = Model(LogisticRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def regression_result(self) -> Result:
        """Setup a regression Result"""
        dataset = load_demo_dataset("boston")
        model = Model(LinearRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, classifier_result: Result) -> Axes:
        """Setup a permutation importance plot"""
        yield classifier_result.plot.permutation_importance()
        plt.close()

    def test_can_be_given_an_ax(self, classifier_result: Result):
        """
        Expect that if the method is given an ax, it will use that ax
        to plot on
        """
        fig, ax = plt.subplots()
        test_ax = classifier_result.plot.feature_importance(ax=ax, class_name=0)
        assert ax == test_ax
        plt.close()

    def test_has_correct_ylabel(self, ax: Axes):
        """Expect that the plot will have a correct ylabel"""
        assert ax.get_ylabel() == "Feature Labels"

    def test_has_correct_xlabel(self, ax: Axes):
        """Expect that the plot will have the correct xlabel"""
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )

    def test_can_turn_labels_off(self, classifier_result: Result):
        """
        Expect classifier results to be able to toggle labels off with the add_label flag
        """
        ax = classifier_result.plot.permutation_importance(add_label=False)
        assert len(ax.texts) == 0

    @pytest.mark.parametrize(
        "top_n, bottom_n, expected",
        [
            (None, None, feature_importances),
            (2, None, feature_importances[:2]),
            (None, 2, feature_importances[2:]),
            (2, 2, feature_importances),
            (0.2, None, feature_importances[:1]),
            (None, 0.2, feature_importances[-1:]),
            (0.2, 0.2, [feature_importances[0], feature_importances[-1]]),
        ],
    )
    def test_has_correct_label_for_each_bar(
        self,
        top_n: Union[int, float],
        bottom_n: Union[int, float],
        expected: List[str],
        classifier_result: Result,
    ):
        """
        Expect that the plot will have the correct labels for each bar
        depending on top_n and bottom_n
        """
        ax = classifier_result.plot.permutation_importance(
            top_n=top_n, bottom_n=bottom_n
        )
        assert {text.get_text() for text in ax.texts} == set(expected)

    @pytest.mark.parametrize(
        "top_n, bottom_n, expected",
        [
            (None, None, "Permutation Importances (Accuracy) - LogisticRegression"),
            (
                2,
                None,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 2",
            ),
            (
                None,
                2,
                "Permutation Importances (Accuracy) - LogisticRegression - Bottom 2",
            ),
            (
                2,
                2,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 2 - Bottom 2",
            ),
            (
                0.2,
                None,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 20%",
            ),
            (
                None,
                0.2,
                "Permutation Importances (Accuracy) - LogisticRegression - Bottom 20%",
            ),
            (
                0.2,
                0.2,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 20% - Bottom "
                "20%",
            ),
            (
                1,
                0.2,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 1 - Bottom 20%",
            ),
            (
                0.2,
                1,
                "Permutation Importances (Accuracy) - LogisticRegression - Top 20% - Bottom 1",
            ),
        ],
    )
    def test_has_correct_title(
        self,
        classifier_result: Result,
        top_n: Union[int, float],
        bottom_n: Union[int, float],
        expected: str,
    ):
        """
        Expect that the plot has a correct title, depending on the top_n and bottom_n
        arguments
        """
        ax = classifier_result.plot.permutation_importance(
            top_n=top_n, bottom_n=bottom_n
        )
        assert ax.title.get_text() == expected

    @pytest.mark.parametrize(
        "top_n, bottom_n, expected",
        [
            (None, None, 4),
            (2, None, 2),
            (1, 1, 2),
            (None, 2, 2),
            (0.2, 0.2, 2),
            (0.2, None, 1),
            (None, 0.2, 1),
            (1, 0.2, 2),
            (0.2, 1, 2),
        ],
    )
    def test_has_correct_number_of_labels(
        self,
        classifier_result: Result,
        top_n: Union[int, float],
        bottom_n: Union[int, float],
        expected: int,
    ):
        """
        Expect that the plot has correct number of labels based on
        top_n and bottom_n parameters
        """
        ax = classifier_result.plot.permutation_importance(
            top_n=top_n, bottom_n=bottom_n
        )
        assert len(ax.texts) == expected
        plt.close()

    def test_plots_have_correct_title_when_using_pipeline(self, dataset: Dataset):
        """
        Expect plots to work correctly with pipelines,
        showing the title of the estimator and not Pipeline
        """
        pipe = Pipeline(
            [
                ("scale", DFStandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10)),
            ]
        )

        model = Model(pipe)
        result = model.score_estimator(dataset)
        ax = result.plot.permutation_importance()

        assert (
            ax.title.get_text()
            == "Permutation Importances (Accuracy) - RandomForestClassifier"
        )

        assert 4 == len(list(ax.get_yticklabels()))
        plt.close()

    def test_regressor_works_as_expected(self, regression_result: Result):
        """Expect that regression results can plot properly and has the correct title"""
        ax = regression_result.plot.permutation_importance()
        assert ax.title.get_text() == "Permutation Importances (R2) - LinearRegression"
        plt.close()

    def test_has_correct_xlabel_when_using_trees(self, dataset: Dataset):
        """Expect plotting feature_importance of a RandomForest to show Feature Importances as
        xlabels instead of coef"""
        model = Model(RandomForestClassifier())
        result = model.score_estimator(dataset)
        ax = result.plot.permutation_importance()
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        plt.close()

    def test_raises_if_func_is_passed_unfitted_estimator(self, dataset: Dataset):
        """Expect the function to raise if an unfitted estimator is passed"""
        with pytest.raises(sklearn.exceptions.NotFittedError):
            plot_permutation_importance(RandomForestClassifier(), dataset.x, dataset.y)
