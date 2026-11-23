"""True, eccentric and mean anomaly of the elliptical orbit; Kepler's equation.

Symbols: ``f`` true anomaly, ``E`` eccentric anomaly, ``M`` mean anomaly, all
in radians. Valid for 0 <= e < 1. Every argument may be one value or an array
of n values.

The conversions keep the revolution count: an anomaly of 2 pi + x is mapped to
2 pi + (the image of x), so angles that grow with time stay continuous.

THE THREE ANOMALIES
-------------------
    f  true anomaly:      the real angle at the focus, from periapsis to the
                          spacecraft. Not uniform in time.
    E  eccentric anomaly: an auxiliary angle at the centre of the ellipse,
                          which makes the geometry simple: r = a (1 - e cos E).
    M  mean anomaly:      M = n (t - t_periapsis), uniform in time; an angle
                          only in name.

    f  <-- geometry -->  E  <-- Kepler's equation -->  M  <-- n -->  time

FIRST READING: ``single_state.M2E_single`` solves Kepler's equation for one
value with an ordinary loop. ``M2E`` here does the same for many values.
"""

import numpy as np

from ._util import TWO_PI, out


def _check_e(e):
    e = np.asarray(e, float)
    if np.any((e < 0.0) | (e >= 1.0)):
        raise ValueError("These relations hold for the ellipse only: 0 <= e < 1.")
    return e


def f2E(f, e):
    """Eccentric anomaly from the true anomaly.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*; for the form used
    here, Broucke and Cefola, "A note on the relations between true and
    eccentric anomalies in the two-body problem" (1973).
    Lecture 3, eccentric anomaly.

    Notes
    -----
    Uses E = f - 2 atan(beta sin f / (1 + beta cos f)), with
    beta = e / (1 + sqrt(1 - e^2)), which has no quadrant ambiguity.
    """
    e = _check_e(e)
    f = np.asarray(f, dtype=float)
    # The textbook formula  tan(E/2) = sqrt((1-e)/(1+e)) tan(f/2)  breaks at
    # f = pi and loses the revolution count. The form below is equivalent
    # and has neither problem: the atan2 term is the (bounded) difference
    # between f and E, which is zero at the apses.
    beta = e / (1.0 + np.sqrt(1.0 - e**2))       # 0 for a circle, -> 1 as e -> 1
    difference = 2.0 * np.arctan2(beta * np.sin(f), 1.0 + beta * np.cos(f))   # = f - E
    return f - difference


def E2f(E, e):
    """True anomaly from the eccentric anomaly.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*; for the form used
    here, Broucke and Cefola, "A note on the relations between true and
    eccentric anomalies in the two-body problem" (1973).
    Lecture 3, eccentric anomaly.

    Notes
    -----
    Uses f = E + 2 atan(beta sin E / (1 - beta cos E)), with
    beta = e / (1 + sqrt(1 - e^2)).
    """
    e = _check_e(e)
    E = np.asarray(E, dtype=float)
    # Mirror image of f2E: the same bounded difference f - E, now written
    # in terms of E.
    beta = e / (1.0 + np.sqrt(1.0 - e**2))
    difference = 2.0 * np.arctan2(beta * np.sin(E), 1.0 - beta * np.cos(E))   # = f - E
    return E + difference


def E2M(E, e):
    """Mean anomaly from the eccentric anomaly: Kepler's equation, M = E - e sin E.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 4, mean anomaly and Kepler's equation.
    """
    e = _check_e(e)
    E = np.asarray(E, dtype=float)
    return E - e * np.sin(E)                     # Kepler's equation, direct sense


def M2E(M, e, tol: float = 1e-13, max_iter: int = 50):
    """Eccentric anomaly from the mean anomaly: numerical solution of Kepler's equation.

    Parameters
    ----------
    M : mean anomaly, radians (any value; the revolution count is kept)
    e : eccentricity, 0 <= e < 1
    tol : the iteration stops when every correction is below ``tol`` radians
    max_iter : maximum number of iterations

    Returns
    -------
    E such that E - e sin E = M

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 4, numerical solution and algorithmic aspects.

    Notes
    -----
    Newton's method on the mean anomaly reduced to [-pi, pi]. Starting value:
    M for e < 0.8, otherwise pi with the sign of M, from which the iteration
    converges for every e < 1. Raises ``RuntimeError`` if ``max_iter`` is
    reached.
    """
    e = _check_e(e)
    # Step 0: common shape (primer 1): one e with many M, or one M with many e.
    M, e = np.broadcast_arrays(np.asarray(M, dtype=float), e)

    # Step 1: remove the whole revolutions, so that M is in [-pi, pi].
    #         They are added back at the end.
    revolutions = np.round(M / TWO_PI)
    M_reduced = M - TWO_PI * revolutions

    # Step 2: starting value, chosen value by value (primer 6):
    #         E = M for moderate e; E = +pi or -pi (the sign of M) for high e.
    E = np.where(e < 0.8, M_reduced, np.pi * np.sign(M_reduced))

    # Step 3: Newton's iteration on  g(E) = E - e sin E - M = 0,
    #         with  g'(E) = 1 - e cos E:    E <- E - g(E) / g'(E).
    #         All values are iterated together and the loop stops when the
    #         LARGEST correction is below tol. Values that converged early
    #         simply receive corrections of about zero.
    for _ in range(max_iter):
        correction = (E - e * np.sin(E) - M_reduced) / (1.0 - e * np.cos(E))
        E = E - correction
        if np.all(np.abs(correction) < tol):
            break
    else:   # runs only if the loop was never broken
        raise RuntimeError("Kepler's equation: no convergence within max_iter iterations.")

    # Step 4: put the whole revolutions back.
    return out(E + TWO_PI * revolutions)


def f2M(f, e):
    """Mean anomaly from the true anomaly (through the eccentric anomaly).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 4, mean anomaly and Kepler's equation.
    """
    return E2M(f2E(f, e), e)


def M2f(M, e, tol: float = 1e-13, max_iter: int = 50):
    """True anomaly from the mean anomaly (solves Kepler's equation).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 4, position as a function of time.
    """
    return E2f(M2E(M, e, tol=tol, max_iter=max_iter), e)


__all__ = ["f2E", "E2f", "E2M", "M2E", "f2M", "M2f"]
