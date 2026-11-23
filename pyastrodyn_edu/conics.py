"""Scalar relations of the Keplerian orbit: shape, speeds, period.

Layer 2 (numerical). Every function works in any consistent unit system.
Every argument may be one value or an array of n values; the result has the
matching shape. Angles are in radians.

HOW TO READ THIS MODULE: each function is one formula, stated in its
documentation string and written in one or two lines of code. Nothing here
depends on the shape of the inputs: NumPy applies the formula to every value
(primer 1). ``np.asarray(x, float)`` only turns a list or an integer into an
array of decimal numbers.

Accuracy: all functions are closed-form and accurate to floating-point
rounding.
"""

import numpy as np

from ._util import dot, norm, out


def orbit_radius(p, e, f):
    """Distance from the focus as a function of the true anomaly (orbit equation).

    Parameters
    ----------
    p : semi-latus rectum
    e : eccentricity
    f : true anomaly, radians

    Returns
    -------
    r = p / (1 + e cos f)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, the orbit equation and the conic sections.

    Notes
    -----
    For e >= 1 the result is physical only where 1 + e cos f > 0.
    """
    return np.asarray(p, float) / (1.0 + np.asarray(e, float) * np.cos(f))


def semilatus_rectum(a, e):
    """Semi-latus rectum, p = a (1 - e^2).

    Valid for the ellipse (a > 0, e < 1) and the hyperbola (a < 0, e > 1).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, the orbit equation and the conic sections.
    """
    return np.asarray(a, float) * (1.0 - np.asarray(e, float) ** 2)


def periapsis_radius(a, e):
    """Periapsis radius, r_p = a (1 - e). Valid for ellipse and hyperbola.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, the orbit equation and the conic sections.
    """
    return np.asarray(a, float) * (1.0 - np.asarray(e, float))


def apoapsis_radius(a, e):
    """Apoapsis radius, r_a = a (1 + e), for the ellipse; infinity for e >= 1.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, the orbit equation and the conic sections.
    """
    a, e = np.asarray(a, float), np.asarray(e, float)
    return out(np.where(e < 1.0, a * (1.0 + e), np.inf))


def visviva_speed(r, a, mu):
    """Speed on a Keplerian orbit from the vis-viva equation.

    Parameters
    ----------
    r : distance from the centre of attraction
    a : semi-major axis (positive for an ellipse, negative for a hyperbola)
    mu : gravitational parameter

    Returns
    -------
    v = sqrt(mu (2/r - 1/a))

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 2, the energy constant and the vis-viva equation.

    Notes
    -----
    Not valid for the parabola (a infinite); use ``escape_speed`` there.
    """
    return np.sqrt(np.asarray(mu, float) * (2.0 / np.asarray(r, float) - 1.0 / np.asarray(a, float)))


def escape_speed(r, mu):
    """Escape (parabolic) speed at distance r, v = sqrt(2 mu / r).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, classification of the orbits by energy and eccentricity.
    """
    return np.sqrt(2.0 * np.asarray(mu, float) / np.asarray(r, float))


def circular_speed(r, mu):
    """Speed on a circular orbit of radius r, v = sqrt(mu / r).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 3, elliptical and circular orbits in detail.
    """
    return np.sqrt(np.asarray(mu, float) / np.asarray(r, float))


def orbit_type(e, tol: float = 1e-10):
    """Name of the conic for a given eccentricity.

    Parameters
    ----------
    e : eccentricity, one value or an array
    tol : half-width of the bands around e = 0 and e = 1

    Returns
    -------
    "circular", "elliptical", "parabolic" or "hyperbolic"; a string for one
    value, an array of strings for many.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 2, classification of the orbits by energy and eccentricity.

    Notes
    -----
    The classification by energy is equivalent: negative specific energy for
    the circle and the ellipse, zero for the parabola, positive for the
    hyperbola.
    """
    e = np.asarray(e, float)
    names = np.select(
        [np.abs(e) < tol, e < 1.0 - tol, np.abs(e - 1.0) <= tol],
        ["circular", "elliptical", "parabolic"],
        default="hyperbolic",
    )
    return str(names) if names.ndim == 0 else names


def effective_potential(r, h, mu):
    """Effective potential energy per unit mass of the radial motion.

    Parameters
    ----------
    r : distance from the centre of attraction
    h : magnitude of the specific angular momentum
    mu : gravitational parameter

    Returns
    -------
    U_eff = h^2 / (2 r^2) - mu / r

    References
    ----------
    Schaub and Junkins, *Analytical Mechanics of Space Systems*.
    Lecture 2, effective potential energy and qualitative analysis of the orbits.
    """
    r = np.asarray(r, float)
    return np.asarray(h, float) ** 2 / (2.0 * r**2) - np.asarray(mu, float) / r


def period(a, mu):
    """Orbital period of an ellipse, T = 2 pi sqrt(a^3 / mu). Requires a > 0.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 3, mean motion and period.
    """
    a = np.asarray(a, float)
    if np.any(a <= 0.0):
        raise ValueError("The period is defined for a > 0 (ellipse) only.")
    return 2.0 * np.pi * np.sqrt(a**3 / np.asarray(mu, float))


def mean_motion(a, mu):
    """Mean motion, n = sqrt(mu / |a|^3), in radians per time unit.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 3, mean motion and period.
    """
    return np.sqrt(np.asarray(mu, float) / np.abs(np.asarray(a, float)) ** 3)


def synchronous_radius(rotation_period, mu):
    """Radius of the circular orbit whose period equals a given rotation period.

    With the sidereal rotation period of the Earth this is the geosynchronous
    radius.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 3, geosynchronous orbit.
    """
    T = np.asarray(rotation_period, float)
    return np.cbrt(np.asarray(mu, float) * T**2 / (4.0 * np.pi**2))


def flight_path_angle(e, f):
    """Flight-path angle from the eccentricity and the true anomaly.

    The flight-path angle is the angle of the velocity above the local
    horizontal (the plane normal to the position): positive from periapsis to
    apoapsis, zero at the apses.

    Returns
    -------
    gamma = atan2(e sin f, 1 + e cos f), radians

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 1, definition of flight-path angle.
    """
    e, f = np.asarray(e, float), np.asarray(f, float)
    return np.arctan2(e * np.sin(f), 1.0 + e * np.cos(f))


def flight_path_angle_rv(r_N, v_N):
    """Flight-path angle from position and velocity.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
        Components in any one basis (the same for both).

    Returns
    -------
    gamma = asin(r . v / (|r| |v|)), radians

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 1, definition of flight-path angle.
    """
    s = dot(r_N, v_N) / (norm(r_N) * norm(v_N))
    return np.arcsin(np.clip(s, -1.0, 1.0))


__all__ = [
    "orbit_radius", "semilatus_rectum", "periapsis_radius", "apoapsis_radius",
    "visviva_speed", "escape_speed", "circular_speed", "orbit_type",
    "effective_potential", "period", "mean_motion", "synchronous_radius",
    "flight_path_angle", "flight_path_angle_rv",
]
