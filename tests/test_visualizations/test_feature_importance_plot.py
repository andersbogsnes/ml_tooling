from typing import List, Union

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset, load_demo_dataset
from ml_tooling.plots import plot_feature_importance
from ml_tooling.result import Result
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import VizError

feature_importances = ["-2.26", "-0.98", "0.82", "-0.54"]


class TestFeatureImportancePlot:
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
        dataset = load_demo_dataset("california")
        model = Model(LinearRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, classifier_result: Result) -> Axes:
        """Setup a feature importance plot"""
        yield classifier_result.plot.feature_importance(class_index=0)
        plt.close()

    def test_can_be_given_an_ax(self, classifier_result: Result):
        """
        Expect that if the method is given an ax, it will use that ax
        to plot on
        """
        fig, ax = plt.subplots()
        test_ax = classifier_result.plot.feature_importance(ax=ax, class_index=0)
        assert ax == test_ax
        plt.close()

    def test_has_correct_ylabel(self, ax: Axes):
        """Expect that the plot will have a correct ylabel"""
        assert ax.get_ylabel() == "Feature Labels"

    def test_has_correct_xlabel(self, ax: Axes):
        """Expect that the plot will have the correct xlabel"""
        assert ax.get_xlabel() == "Coefficients"

    def test_can_turn_labels_off(self, classifier_result: Result):
        """
        Expect classifier results to be able to toggle labels off with the add_label flag
        """
        ax = classifier_result.plot.feature_importance(add_label=False, class_index=0)
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
        Expect that feature_importance plot will have the correct labels for each bar
        depending on top_n and bottom_n
        """
        ax = classifier_result.plot.feature_importance(
            top_n=top_n, bottom_n=bottom_n, class_index=0
        )
        assert {text.get_text() for text in ax.texts} == set(expected)

    @pytest.mark.parametrize(
        "top_n, bottom_n, expected",
        [
            (None, None, "Feature Importances - LogisticRegression"),
            (2, None, "Feature Importances - LogisticRegression - Top 2"),
            (None, 2, "Feature Importances - LogisticRegression - Bottom 2"),
            (2, 2, "Feature Importances - LogisticRegression - Top 2 - Bottom 2"),
            (0.2, None, "Feature Importances - LogisticRegression - Top 20%"),
            (None, 0.2, "Feature Importances - LogisticRegression - Bottom 20%"),
            (
                0.2,
                0.2,
                "Feature Importances - LogisticRegression - Top 20% - Bottom 20%",
            ),
            (1, 0.2, "Feature Importances - LogisticRegression - Top 1 - Bottom 20%"),
            (0.2, 1, "Feature Importances - LogisticRegression - Top 20% - Bottom 1"),
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
        ax = classifier_result.plot.feature_importance(
            top_n=top_n, bottom_n=bottom_n, class_index=0
        )
        assert ax.title.get_text() == expected
        plt.close()

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
        Expect that feature_importance has correct number of labels based on
        top_n and bottom_n parameters
        """
        ax = classifier_result.plot.feature_importance(
            top_n=top_n, bottom_n=bottom_n, class_index=0
        )
        assert len(ax.texts) == expected
        plt.close()

    def test_plots_have_correct_title_when_using_pipeline(self, dataset: Dataset):
        """
        Expect feature importance plots to work correctly with pipelines,
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
        ax = result.plot.feature_importance()

        assert ax.title.get_text() == "Feature Importances - RandomForestClassifier"

        assert 4 == len(list(ax.get_yticklabels()))
        plt.close()

    def test_regressor_works_as_expected(self, regression_result: Result):
        """Expect that regression results can plot properly and has the correct title"""
        ax = regression_result.plot.feature_importance()
        assert ax.title.get_text() == "Feature Importances - LinearRegression"
        plt.close()

    def test_has_correct_xlabel_when_using_trees(self, dataset: Dataset):
        """Expect plotting feature_importance of a RandomForest to show Feature Importances as
        xlabels instead of coef"""
        model = Model(RandomForestClassifier())
        result = model.score_estimator(dataset)
        ax = result.plot.feature_importance()
        assert ax.get_xlabel() == "Feature Importances"
        plt.close()

    def test_has_correct_title_when_using_trees(self, dataset: Dataset):
        """Expect the plot to not have Class in the title"""
        model = Model(RandomForestClassifier())
        result = model.score_estimator(dataset)

        ax = result.plot.feature_importance(class_index=10)
        assert "Class 10" not in ax.title.get_text()
        plt.close()

    def test_raises_if_passed_model_without_feature_importance_or_coefs(
        self, dataset: Dataset
    ):
        """
        Expect an exception if trying to plot an estimator that doesn't have
        coefficients or feature_importance
        """
        model = Model(KNeighborsClassifier())
        result = model.score_estimator(dataset)

        with pytest.raises(VizError):
            result.plot.feature_importance()

    def test_raises_if_func_is_passed_unfitted_estimator(self, dataset: Dataset):
        """Expect the function to raise if an unfitted estimator is passed"""
        with pytest.raises(VizError):
            plot_feature_importance(RandomForestClassifier(), dataset.x, dataset.y)

    def test_raises_if_passed_an_invalid_classname(self, classifier_result: Result):
        """Expect plot to raise if trying to access a class_index that doesn't exist"""
        with pytest.raises(VizError):
            classifier_result.plot.feature_importance(class_index=100)
