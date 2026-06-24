"""Utility functions for visualization."""

import numpy as np
from matplotlib.patches import Circle as _Circle, Ellipse as _Ellipse
from matplotlib.pyplot import Axes as _PLTAxes


def draw_ellipse_around_points(p1, p2, ax, b_ratio=0.15):
    """Draw an ellipse around the specified points.

    Draw an ellipse containing p1 and p2 located over the main axis,
    symmetrically around the center.
    """
    # Compute center
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

    # Compute major axis length (2a) and angle
    dx, dy = x2 - x1, y2 - y1
    a = np.hypot(dx, dy) / 2
    angle = np.degrees(np.arctan2(dy, dx))  # degrees for matplotlib

    # Minor axis length
    b = a * b_ratio

    # Plotting
    ellipse = _Ellipse(
        (xc, yc),
        width=3 * a,
        height=2.1 * b,
        angle=angle,
        edgecolor=(0, 0, 1, 0.2),
        facecolor=(0, 0.3, 1, 0.2),
        lw=2,
    )

    ax.add_patch(ellipse)


def draw_operator(op, axis: _PLTAxes) -> _PLTAxes:
    """Draw the operator op over the axis.

    Parameters
    ----------
    op: Operator
      If the operator acts on a single site, draws a disk on its coordinates.
      If is a SumOperator, flatten it and draw each term.
      For many-body operators, a line is drawn.
    axis: mpl.Axis
      the axis over which the operator is going to be drawn.

    Return
    ------
    mpl.Axis
      the axis over which the operator was drawn.

    """
    # TODO: handle 3D graphs
    from qalma.operators import SumOperator

    system = op.system
    g = system.spec["graph"]
    g.complete_coordiantes()
    op = op.flat()
    if isinstance(op, SumOperator):
        for term in op.terms:
            draw_operator(term, axis)
        return axis
    acts_over = op.acts_over()
    if acts_over is not None:
        coords = [g.nodes[site]["coords"] for site in acts_over]
        coords = [(x[0], 0) if len(x) == 1 else x for x in coords]
        if len(coords) == 1:
            axis.add_artist(_Circle(coords[0], 0.1))
        if len(coords) == 2:
            draw_ellipse_around_points(coords[0], coords[1], axis)
        else:
            axis.plot(
                [x[0] for x in coords] + [coords[0][0]],
                [x[1] for x in coords] + [coords[0][1]],
                lw="5",
                c="red",
            )
    return axis
