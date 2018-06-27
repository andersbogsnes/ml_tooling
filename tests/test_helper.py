import matplotlib.pyplot as plt
from ml_utils.visualizations.helpers import generate_text_labels
from ml_utils.metrics import lift_score
import numpy as np


def test_add_text_labels_vertical_returns_correct():
    fig, ax = plt.subplots()
    ax.bar(['value'], [100])
    x_values, y_values = next(generate_text_labels(ax, horizontal=False))
    assert 0 == x_values
    assert (100 + 105 * .005) == y_values


def test_add_text_labels_horizontal_returns_correct():
    fig, ax = plt.subplots()
    ax.barh(['value'], [100])
    x_values, y_values = next(generate_text_labels(ax, horizontal=True))
    assert 0 == y_values
    assert (100 + 105 * .005) == x_values


def test_lift_score_returns_correctly():
    y_targ = np.array([1, 1, 1, 0, 0, 2, 0, 3, 4])
    y_pred = np.array([1, 0, 1, 0, 0, 2, 1, 3, 0])

    result = lift_score(y_targ, y_pred)
    assert 2 == result
