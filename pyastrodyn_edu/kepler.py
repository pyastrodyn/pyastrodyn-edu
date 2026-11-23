"""Position as a function of time on an elliptical orbit.

Valid for 0 <= e < 1. Any consistent unit system; angles in radians. Every
argument may be one value or an array of n values.
"""

import numpy as np

from ._util import TWO_PI
from .anomalies import M2f, f2M
from .conics import mean_motion
from .elements import coe2rv, rv2coe


def time_since_periapsis(f, a, e, mu):
    """Time elapsed since the last periapsis passage.

    Parameters
    ----------
    f : true anomaly, radians
    a : semi-major axis (a > 0)
    e : eccentricity (0 <= e < 1)
    mu : gravitational parameter

    Returns
    -------
    Time in [0, period).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 4, the epoch and the sixth integral of motion.
    """
    M = np.mod(f2M(f, e), TWO_PI)                # mean anomaly since the LAST periapsis
    n = mean_motion(a, mu)                       # radians per time unit
    return M / n                                 # because M = n (t - t_periapsis)


def propagate_kepler(a, e, f0, dt, mu):
    """True anomaly after a time interval, by Kepler's equation.

    On a Keplerian orbit five elements are constant; only the true anomaly
    changes with time. This function returns its new value.

    Parameters
    ----------
    a : semi-major axis (a > 0)
    e : eccentricity (0 <= e < 1)
    f0 : true anomaly at the start, radians
    dt : time interval (may be negative)
    mu : gravitational parameter

    Returns
    -------
    f : true anomaly at the end, radians. Not reduced to [0, 2 pi): it grows
        by 2 pi per revolution, so it is continuous in ``dt``.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 4, position as a function of time.
    """
    # Step 1: from the true anomaly to the mean anomaly at the start.
    M0 = f2M(f0, e)
    # Step 2: the mean anomaly grows uniformly:  M = M0 + n dt.
    M = M0 + mean_motion(a, mu) * np.asarray(dt, dtype=float)
    # Step 3: back to the true anomaly (this is where Kepler's equation is solved).
    return M2f(M, e)


def propagate_kepler_rv(r0_N, v0_N, dt, mu):
    """Position and velocity after a time interval, by Kepler's equation.

    The state is converted to elements, the true anomaly is advanced with
    ``propagate_kepler``, and the elements are converted back.

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,) or (n, 3)
        State at the start, components in N.
    dt : float or ndarray, shape (m,)
        Time interval. One state may be combined with m intervals.
    mu : gravitational parameter

    Returns
    -------
    r_N, v_N : ndarray, shape (3,), (n, 3) or (m, 3)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 4, position as a function of time.
    """
    # Step 1: state -> elements. Five of them are constants of the motion.
    a, e, i, raan, argp, f0 = rv2coe(r0_N, v0_N, mu)
    # Step 2: advance the only element that changes, the true anomaly.
    f = propagate_kepler(a, e, f0, dt, mu)
    # Step 3: elements -> state. With one initial state and m intervals,
    #         f has shape (m,) and the result has shape (m, 3) (primer 1).
    return coe2rv(a, e, i, raan, argp, f, mu)


__all__ = ["time_since_periapsis", "propagate_kepler", "propagate_kepler_rv"]
