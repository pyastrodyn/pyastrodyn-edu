"""Lagrange coefficients: the state at a later point from the initial state.

On a Keplerian orbit the position and velocity at any point are linear
combinations of the initial position and velocity,

    r = F r0 + G v0,        v = Fdot r0 + Gdot v0.

Here the coefficients are written in terms of the change of true anomaly,
``df``. Valid for every conic. Any consistent unit system; angles in radians.
"""

import numpy as np

from ._util import col, dot, norm, out


def lagrange_coefficients(r0_N, v0_N, df, mu):
    """Lagrange coefficients for a change ``df`` of the true anomaly.

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,) or (n, 3)
        Initial position and velocity (components in any one basis).
    df : float or ndarray
        Change of true anomaly, radians. One state may be combined with m
        values of ``df``.
    mu : gravitational parameter

    Returns
    -------
    F, G, Fdot, Gdot : float or ndarray

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 5, Lagrange coefficients.

    Notes
    -----
    The four coefficients satisfy ``F * Gdot - G * Fdot = 1`` (conservation
    of the angular momentum). The expression used for ``Fdot`` has no
    division by ``sin(df)``, so it is regular for every ``df``.
    """
    r0_N = np.asarray(r0_N, dtype=float)
    v0_N = np.asarray(v0_N, dtype=float)
    df = np.asarray(df, dtype=float)

    # Step 1: three scalars of the initial state.
    r0 = norm(r0_N)                              # initial distance
    radial_speed0 = dot(r0_N, v0_N) / r0         # initial radial velocity
    h = norm(np.cross(r0_N, v0_N))               # angular momentum (constant)
    p = h**2 / mu                                # semi-latus rectum (constant)
    cos_df, sin_df = np.cos(df), np.sin(df)

    # Step 2: distance at the new point. It is the orbit equation
    #         r = p / (1 + e cos f)  with  f = f0 + df  expanded, and with
    #         e cos f0 = p/r0 - 1  and  e sin f0 = h radial_speed0 / mu
    #         taken from the initial state. No element is needed.
    e_cos_f0 = p / r0 - 1.0
    e_sin_f0 = h * radial_speed0 / mu
    r = p / (1.0 + e_cos_f0 * cos_df - e_sin_f0 * sin_df)

    # Step 3: the four coefficients.
    F = 1.0 - (r / p) * (1.0 - cos_df)
    G = r * r0 * sin_df / h
    Fdot = (mu / h**2) * radial_speed0 * (1.0 - cos_df) - (mu / (h * r0)) * sin_df
    Gdot = 1.0 - (r0 / p) * (1.0 - cos_df)
    return out(F), out(G), out(Fdot), out(Gdot)


def propagate_lagrange(r0_N, v0_N, df, mu):
    """Position and velocity after a change ``df`` of the true anomaly.

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,) or (n, 3)
    df : float or ndarray, shape (m,)
    mu : gravitational parameter

    Returns
    -------
    r_N, v_N : ndarray, shape (3,), (n, 3) or (m, 3)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 5, Lagrange coefficients.
    """
    r0_N = np.asarray(r0_N, dtype=float)
    v0_N = np.asarray(v0_N, dtype=float)
    F, G, Fdot, Gdot = lagrange_coefficients(r0_N, v0_N, df, mu)

    # r = F r0 + G v0   and   v = Fdot r0 + Gdot v0.
    # Each coefficient is a scalar (one per state, or one per value of df)
    # that multiplies a vector, hence col() (primer 3). With ONE initial
    # state and m values of df the result has shape (m, 3): row k is the
    # state after df[k].
    r_N = col(F) * r0_N + col(G) * v0_N
    v_N = col(Fdot) * r0_N + col(Gdot) * v0_N
    return r_N, v_N


__all__ = ["lagrange_coefficients", "propagate_lagrange"]
