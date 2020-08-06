import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.result import Result
from ml_tooling.utils import VizError


class TestRocCurve:
    @pytest.fixture(scope="class")
    def classifier_result(self) -> Result:
        """Setup a classifier Result"""
        dataset = load_demo_dataset("iris")
        model = Model(LogisticRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, classifier_result: Result) -> Axes:
        """Setup a roc_curve plot"""
        yield classifier_result.plot.roc_curve()
        plt.close()

    def test_roc_curve_plots_can_be_given_an_ax(self, classifier_result: Result):
        """Expect the plot to be drawn on the ax provided"""
        fig, ax = plt.subplots()
        test_ax = classifier_result.plot.roc_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_title(self, ax: Axes):
        """Expect the plot to have the correct title"""
        assert ax.title.get_text() == "ROC AUC - LogisticRegression"

    def test_has_correct_ylabel(self, ax: Axes):
        """Expect the plot to have the correct y label"""
        assert ax.get_ylabel() == "True Positive Rate"

    def test_has_correct_xlabel(self, ax: Axes):
        """Expect the plot to have the correct x label"""
        assert ax.get_xlabel() == "False Positive Rate"

    @pytest.mark.parametrize("class_index", [0, 1, 2])
    def test_roc_curve_have_correct_data(self,
                                         ax: Axes,
                                         classifier_result: Result,
                                         class_index: int):
        """Expect the plot to have the correct data"""
        x = classifier_result.data.test_x
        y = classifier_result.data.test_y
        y_true = label_binarize(y, classes=np.unique(y))[:, class_index]

        y_proba = classifier_result.estimator.predict_proba(x)[:, class_index]
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        assert np.all(fpr == ax.lines[class_index].get_xdata())
        assert np.all(tpr == ax.lines[class_index].get_ydata())
        plt.close()

    def test_roc_curve_fails_correctly_without_predict_proba(self):
        dataset = load_demo_dataset("iris")
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(dataset)
        with pytest.raises(VizError):
            result.plot.roc_curve()
        plt.close()
