import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.plots.viz import RegressionVisualize, ClassificationVisualize
from ml_tooling.result import Result


class TestVisualize:
    @pytest.fixture(scope="class")
    def regression_result(self) -> Result:
        """Setup a regression Result"""
        dataset = load_demo_dataset("boston")
        model = Model(LinearRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def classification_result(self) -> Result:
        """Setup a classification Result"""
        dataset = load_demo_dataset("iris")
        model = Model(LogisticRegression())
        return model.score_estimator(dataset)

    def test_result_regression_gets_correct_visualizers(
        self, regression_result: Result
    ):
        """Expect a regression result to have a RegressionVisualize class when plotting"""
        assert isinstance(regression_result.plot, RegressionVisualize)

    def test_result_classification_gets_correct_visualizers(
        self, classification_result: Result
    ):
        """Expect a classification result to have a ClassificatoinVisualize class when plotting"""
        assert isinstance(classification_result.plot, ClassificationVisualize)

    @pytest.mark.parametrize(
        "attr",
        ["residuals", "prediction_error", "feature_importance", "learning_curve"],
    )
    def test_regression_visualize_has_all_plots(
        self, attr: str, regression_result: Result
    ):
        """Expect a regression Result to be able to plot all the given plots"""
        plotter = getattr(regression_result.plot, attr)()
        assert isinstance(plotter, Axes)
        plt.close()

    @pytest.mark.parametrize(
        "attr",
        [
            "confusion_matrix",
            "roc_curve",
            "lift_curve",
            "feature_importance",
            "pr_curve",
            "learning_curve",
            "permutation_importance",
        ],
    )
    def test_classifier_visualize_has_all_plots(
        self, attr: str, classification_result: Result
    ):
        """Expect a classification Result to be able to plot all the above plots"""
        plotter = getattr(classification_result.plot, attr)()
        assert isinstance(plotter, Axes)
        plt.close()
