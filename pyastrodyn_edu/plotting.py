"""Plots of orbits. A consumer of arrays: no computing module imports this one.

Needs Matplotlib, which is imported only when a plotting function is called.
Each function returns the figure and the axes, so the plot can be modified.
"""

import numpy as np

from .conics import orbit_radius, semilatus_rectum


def plot_orbit_plane(a, e, argp=0.0, ax=None, body_radius=None, num: int = 361, **line_options):
    """Draw one elliptical orbit in its own plane, focus at the origin.

    Parameters
    ----------
    a, e : semi-major axis and eccentricity (e < 1)
    argp : angle from the horizontal axis to the periapsis, radians
    ax : Matplotlib axes to draw on; a new figure is made if omitted
    body_radius : if given, the central body is drawn as a disc of this radius
    num : number of points along the orbit
    line_options : passed to ``ax.plot`` (label, color, linestyle, ...)

    Returns
    -------
    fig, ax

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, the orbit equation and the conic sections.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    f = np.linspace(0.0, 2.0 * np.pi, num)
    r = orbit_radius(semilatus_rectum(a, e), e, f)
    ax.plot(r * np.cos(f + argp), r * np.sin(f + argp), **line_options)
    if body_radius is not None:
        ax.add_patch(plt.Circle((0.0, 0.0), body_radius, color="0.8", zorder=0))
    ax.set_aspect("equal")
    return ax.figure, ax


def plot_orbit_3d(r_N, ax=None, **line_options):
    """Draw a trajectory in three dimensions.

    Parameters
    ----------
    r_N : ndarray, shape (m, 3)
        Positions along the trajectory, components in N.
    ax : Matplotlib 3D axes to draw on; a new figure is made if omitted
    line_options : passed to ``ax.plot``

    Returns
    -------
    fig, ax

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 5, the orbit in three dimensions.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    r_N = np.asarray(r_N, float)
    ax.plot(r_N[:, 0], r_N[:, 1], r_N[:, 2], **line_options)
    ax.set_box_aspect((1, 1, 1))
    return ax.figure, ax


__all__ = ["plot_orbit_plane", "plot_orbit_3d"]
