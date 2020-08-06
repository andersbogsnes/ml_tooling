import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.result import Result
from ml_tooling.utils import VizError


class TestPRCurve:
    @pytest.fixture(scope="class")
    def classifier_result(self) -> Result:
        """Setup a classiifer Result"""
        dataset = load_demo_dataset("iris")
        model = Model(LogisticRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, classifier_result: Result) -> Axes:
        """Setup a PR Curve plot"""
        yield classifier_result.plot.precision_recall_curve()
        plt.close()

    def test_plots_can_be_given_an_ax(self, classifier_result: Result):
        """Expect a plot to be able to be passed an existing axis and plot on that"""
        fig, ax = plt.subplots()
        test_ax = classifier_result.plot.precision_recall_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_the_correct_title(self, ax: Axes):
        """Expect the title to reflect the estimator used"""
        assert ax.title.get_text() == "Precision-Recall - LogisticRegression"

    def test_has_the_correct_ylabel(self, ax: Axes):
        """Expect the plot to have the correct y label"""
        assert ax.get_ylabel() == "Precision"

    def test_has_the_correct_xlabel(self, ax: Axes):
        """Expect the plot to have the correct x label"""
        assert ax.get_xlabel() == "Recall"

    @pytest.mark.parametrize("class_index", [0, 1, 2])
    def test_pr_curve_have_correct_data(
        self, ax: Axes, classifier_result: Result, class_index
    ):
        """Expect the pr curve to have the correct data"""
        x = classifier_result.plot._data.test_x
        y_true = label_binarize(classifier_result.plot._data.test_y, classes=[0, 1, 2])[
            :, class_index
        ]
        y_proba = classifier_result.estimator.predict_proba(x)[:, class_index]

        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        assert np.all(recall == ax.lines[class_index].get_xdata())
        assert np.all(precision == ax.lines[class_index].get_ydata())
        plt.close()

    def test_pr_curve_fails_correctly_without_predict_proba(self):
        """
        Expect that the plot will raise an exception if the estimator
        does not have a predict_proba method
        """
        dataset = load_demo_dataset("iris")
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(dataset)
        with pytest.raises(VizError):
            result.plot.precision_recall_curve()
        plt.close()
