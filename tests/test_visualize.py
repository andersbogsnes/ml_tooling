import pytest
import numpy as np
from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

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


@pytest.mark.parametrize('attr', ['confusion_matrix',
                                  'roc_curve',
                                  'lift_curve',
                                  'feature_importance'])
def test_classifier_visualize_has_all_plots(attr, classifier):
    result = classifier.result.plot
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
    assert expected == {text._text for text in ax.texts}
    assert 'Feature Importance - LogisticRegression' == ax.title._text
    assert 'Features' == ax.get_ylabel()
    assert 'Importance' == ax.get_xlabel()


def test_feature_importance_plots_have_no_labels_if_value_is_false(classifier):
    ax = classifier.result.plot.feature_importance(values=False)
    assert 0 == len(ax.texts)
    assert 'Features' == ax.get_ylabel()
    assert 'Importance' == ax.get_xlabel()
    assert 'Feature Importance - LogisticRegression' == ax.title._text


def test_lift_curve_have_correct_data(classifier):
    ax = classifier.result.plot.lift_curve()

    assert 'Lift Curve - LogisticRegression' == ax.title._text
    assert 'Lift' == ax.get_ylabel()
    assert '% of Data' == ax.get_xlabel()
    assert pytest.approx(19.5) == np.sum(ax.lines[0].get_xdata())
    assert pytest.approx(48.67, rel=.0001) == np.sum(ax.lines[0].get_ydata())


def test_prediction_error_plots_have_correct_data(regression):
    ax = regression.result.plot.prediction_error()
    x, y = regression.result.plot._test_x, regression.result.plot._test_y
    y_pred = regression.result.model.predict(x)

    assert 'Prediction Error - LinearRegression' == ax.title._text
    assert '$\hat{y}$' == ax.get_ylabel()
    assert '$y$' == ax.get_xlabel()

    assert (y_pred == ax.collections[0].get_offsets()[:, 1]).all()
    assert (y == ax.collections[0].get_offsets()[:, 0]).all()


def test_residual_plots_have_correct_data(regression):
    ax = regression.result.plot.residuals()
    x, y = regression.result.plot._test_x, regression.result.plot._test_y
    y_pred = regression.result.model.predict(x)
    expected = y_pred - y

    assert 'Residual Plot - LinearRegression' == ax.title._text
    assert 'Residuals' == ax.get_ylabel()
    assert 'Predicted Value' == ax.get_xlabel()

    assert (expected == ax.collections[0].get_offsets()[:, 1]).all()
    assert (y_pred == ax.collections[0].get_offsets()[:, 0]).all()


def test_roc_curve_have_correct_data(classifier):
    ax = classifier.result.plot.roc_curve()
    x, y = classifier.result.plot._test_x, classifier.result.plot._test_y
    y_proba = classifier.model.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)

    assert 'ROC AUC - LogisticRegression' == ax.title._text
    assert 'True Positive Rate' == ax.get_ylabel()
    assert 'False Positive Rate' == ax.get_xlabel()
    assert (fpr == ax.lines[0].get_xdata()).all()
    assert (tpr == ax.lines[0].get_ydata()).all()


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


def test_viz_get_feature_importance_returns_feature_importance_from_classifier(base):
    classifier = base(RandomForestClassifier())
    result = classifier.score_model()
    viz = ClassificationVisualize(classifier.model,
                                  classifier.config,
                                  classifier.x,
                                  classifier.y,
                                  classifier.x,
                                  classifier.y)
    importance = viz._get_feature_importance()
    assert np.all(result.model.feature_importances_ == importance)


def test_viz_get_labels_returns_array_if_there_are_no_columns(regression):
    input_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    viz = RegressionVisualize(regression.model,
                              regression.config,
                              input_data,
                              regression.y,
                              input_data,
                              regression.y)
    labels = viz._get_labels()
    assert np.all(np.arange(input_data.shape[1]) == labels)
