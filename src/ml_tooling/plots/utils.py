class VizError(Exception):
    """Base Exception for visualization errors"""


def _generate_text_labels(ax, horizontal=False):
    """
    Helper for generating text labels for bar charts

    :param ax:
        Ax which has patches on it

    :param horizontal:
        Whether or not the graph is a horizontal bar chart or a regular bar chart

    :param padding:
        How much padding to multiply by

    :return:
        x and y values for ax.text
    """
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()

        if horizontal is True:
            x_value = width
            y_value = y + (height / 2)
        else:
            x_value = x + (width / 2)
            y_value = height

        yield x_value, y_value