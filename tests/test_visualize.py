import pytest
from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from ml_utils.visualizations import plot_lift_chart
from ml_utils.visualizations.visualizations import (RegressionVisualize,
                                                    ClassificationVisualize,
                                                    VizError)
from sklearn.svm import SVC


def test_result_regression_gets_correct_visualizers(regression):
    result = regression.result
    assert isinstance(result.plot, RegressionVisualize)


def test_result_classification_gets_correct_visualizers(classifier):
    result = classifier.result
    assert isinstance(result.plot, ClassificationVisualize)


@pytest.mark.parametrize('attr', ['residuals', 'prediction_error', 'feature_importance'])
def test_regression_visualize_has_all_plots(attr, regression):
    result = regression.result.plot
    plotter = getattr(result, attr)()
    assert isinstance(plotter, Axes)


@pytest.mark.parametrize('attr', ['confusion_matrix',
                                  'roc_curve',
                                  'lift_curve',
                                  'feature_importance'])
def test_classifier_visualize_has_all_plots(attr, classifier):
    result = classifier.result.plot
    plotter = getattr(result, attr)()
    assert isinstance(plotter, Axes)


def test_roc_curve_fails_correctly_without_predict_proba(base):
    svc = base(SVC())
    result = svc.score_model()
    with pytest.raises(VizError):
        result.plot.roc_curve()


def test_feature_importance_fails_correctly_without_predict_proba(base):
    svc = base(SVC())
    result = svc.score_model()
    with pytest.raises(VizError):
        result.plot.feature_importance()


def test_lift_chart_fails_correctly_with_2d_proba():
    x, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(x, y)
    proba = clf.predict_proba(x)
    with pytest.raises(VizError):
        plot_lift_chart(y, proba)
