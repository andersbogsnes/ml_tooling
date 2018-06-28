import matplotlib.pyplot as plt
from ml_utils.visualizations.helpers import generate_text_labels


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


