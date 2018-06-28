import pytest
import numpy as np
from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from ml_utils.visualizations import plot_lift_chart
from ml_utils.visualizations.visualizations import (RegressionVisualize,
                                                    ClassificationVisualize,
                                                    VizError)
from sklearn.svm import SVC

np.random.seed(42)


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


def test_confusion_matrix_plots_have_correct_data(classifier):
    ax = classifier.result.plot.confusion_matrix()

    assert 'Confusion Matrix - LogisticRegression - Normalized' == ax.title._text
    result = {text._text for text in ax.texts}
    assert {'0.16', '0.38', '0.62', '0.84'} == result
    assert 'True Label' == ax.get_ylabel()
    assert 'Predicted Label' == ax.get_xlabel()


def test_feature_importance_plots_have_correct_data(classifier):
    ax = classifier.result.plot.feature_importance()

    expected = {'-1.76', '-0.83', '0.34', '0.60'}
    assert 'Feature Importance - LogisticRegression' == ax.title._text
    assert expected == {text._text for text in ax.texts}
    assert 'Features' == ax.get_ylabel()
    assert 'Importance' == ax.get_xlabel()


def test_lift_curve_have_correct_data(classifier):
    ax = classifier.result.plot.lift_curve()

    assert 'Lift Curve - LogisticRegression' == ax.title._text
    assert 'Lift' == ax.get_ylabel()
    assert '% of Data' == ax.get_xlabel()
    assert pytest.approx(19.5) == np.sum(ax.lines[0].get_xdata())
    assert pytest.approx(48.67, rel=.0001) == np.sum(ax.lines[0].get_ydata())

def test

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


def test_viz_get_feature_importance_returns_coef_from_regression(regression):
    viz = RegressionVisualize(regression.model,
                              regression.config,
                              regression.x,
                              regression.y,
                              regression.x,
                              regression.y)
    importance = viz._get_feature_importance()
    assert np.all(regression.model.coef_ == importance)
