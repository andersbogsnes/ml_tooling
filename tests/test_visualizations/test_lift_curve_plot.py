import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.result import Result
from ml_tooling.utils import VizError


class TestLiftCurvePlot:
    @pytest.fixture(scope="class")
    def iris_result(self) -> Result:
        dataset = load_demo_dataset("iris")
        model = Model(LogisticRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, iris_result: Result) -> Axes:
        """Setup an Ax with a drawing of the plot"""
        yield iris_result.plot.lift_curve()
        plt.close()

    @pytest.fixture(scope="class")
    def ax_labels(self, iris_result: Result) -> Axes:
        """Setup an Ax with a drawing of the plot including labels"""
        yield iris_result.plot.lift_curve(labels=["Setosa", "Versicolor", "Virginica"])
        plt.close()

    def test_lift_curve_plots_can_be_given_an_ax(self, iris_result: Result):
        """Expect a plot to use a given axes if passed as an argument"""
        fig, ax = plt.subplots()
        test_ax = iris_result.plot.lift_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_lift_curve_has_correct_title(self, ax: Axes):
        """Expect the lift curve to have a title matching the input model"""
        assert "Lift Curve - LogisticRegression" == ax.title.get_text()

    def test_lift_curve_has_correct_ylabel(self, ax: Axes):
        """Expect the lift curve to have the same label"""
        assert "Lift" == ax.get_ylabel()

    def test_lift_curve_has_correct_xlabel(self, ax: Axes):
        """Expect the lift curve to have the same xlabel"""
        assert "% of Data" == ax.get_xlabel()

    def test_lift_curve_has_correct_number_of_lines(self, ax: Axes):
        """Expect the lift curve to have one line per class + baseline"""
        assert len(ax.lines) == 4

    @pytest.mark.parametrize(
        "line_no, value", [(0, 19.5), (1, 19.5), (2, 19.5), (3, 1)]
    )
    def test_lift_curve_to_have_the_correct_xdata(
        self, ax: Axes, line_no: int, value: float
    ):
        """Expect the lift_curve to have the correct x values for each line"""
        assert pytest.approx(value) == np.sum(ax.lines[line_no].get_xdata())

    @pytest.mark.parametrize(
        "line_no, value", [(0, 80.7382), (1, 77.59), (2, 77.59), (3, 2)]
    )
    def test_lift_curve_to_have_the_correct_ydata(
        self, ax: Axes, line_no: int, value: float
    ):
        """Expect the lift_curve to have the correct sum of y values for each line"""
        assert np.sum(ax.lines[line_no].get_ydata()) == pytest.approx(value, rel=0.0001)

    def test_legend_uses_class_numbers_when_no_labels_are_passed(self, ax: Axes):
        """Expect the legend to default to numbers if no labels are passed"""
        expected = {
            "Class 0 $Lift = 3.17$ ",
            "Class 1 $Lift = 2.70$ ",
            "Class 2 $Lift = 2.70$ ",
            "Baseline",
        }
        result = {text.get_text() for text in ax.get_legend().get_texts()}
        assert result == expected

    def test_legend_is_labelled_correctly_when_labels_are_passed(self, ax_labels: Axes):
        """Expect the legend to include labels"""
        expected = {
            "Class Setosa $Lift = 3.17$ ",
            "Class Versicolor $Lift = 2.70$ ",
            "Class Virginica $Lift = 2.70$ ",
            "Baseline",
        }
        result = {text.get_text() for text in ax_labels.get_legend().get_texts()}
        assert result == expected

    def test_raises_if_wrong_number_of_labels_is_passed(self, iris_result: Result):
        """Expect lift_curve to fail if wrong number of labels is passed"""
        with pytest.raises(
            VizError, match="Number of labels must match number of classes: "
                            "got 1 labels and 3 classes"
        ):
            iris_result.plot.lift_curve(labels=["Only one"])
