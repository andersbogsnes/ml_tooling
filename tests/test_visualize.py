import pytest
from matplotlib.axes import Axes
from ml_utils.visualizations._visualizations import RegressionVisualize, ClassificationVisualize, \
    VizError
from sklearn.svm import SVC


def test_result_regression_gets_correct_visualizers(regression):
    result = regression.result
    assert isinstance(result.plot, RegressionVisualize)


def test_result_classification_gets_correct_visualizers(classifier):
    result = classifier.result
    assert isinstance(result.plot, ClassificationVisualize)


@pytest.mark.parametrize('attr', ['residuals', 'prediction_error'])
def test_regression_visualize_has_all_plots(attr, regression):
    result = regression.result.plot
    plotter = getattr(result, attr)()
    assert isinstance(plotter, Axes)


@pytest.mark.parametrize('attr', ['confusion_matrix', 'roc_curve'])
def test_classifier_visualize_has_all_plots(attr, classifier):
    result = classifier.result.plot
    plotter = getattr(result, attr)()
    assert isinstance(plotter, Axes)


def test_classification_fails_correctly_without_predict_proba(base):
    svc = base(SVC())
    result = svc.test_model()
    with pytest.raises(VizError):
        result.plot.roc_curve()
