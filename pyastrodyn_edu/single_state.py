"""Single-state twins: the hardest algorithms, written for ONE state.

These functions do exactly what ``coe2rv``, ``rv2coe`` and ``M2E`` do, but for
a single state only, with ordinary ``if`` statements and a ``while``-style
loop, and with no array technique at all. They are meant to be READ: start
here, then read the many-states versions, which follow the same steps in the
same order.

They are also a check: the tests require each twin to agree with its
many-states version on regular and on singular cases.

For computing, use the many-states versions; they also accept one state.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def coe2rv_single(a, e, i, raan, argp, f, mu):
    """Position and velocity from the classical elements, for one state.

    Parameters
    ----------
    a, e, i, raan, argp, f : float
        Classical elements; angles in radians.
    mu : float
        Gravitational parameter.

    Returns
    -------
    r_N, v_N : ndarray, shape (3,)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, conversion between the state and the orbital elements.
    """
    # Step 1: semi-latus rectum.
    p = a * (1.0 - e**2)
    if p <= 0.0:
        raise ValueError("a (1 - e^2) must be positive.")

    # Step 2: distance from the orbit equation.
    r = p / (1.0 + e * math.cos(f))

    # Step 3: position and velocity in the perifocal basis P.
    speed_scale = math.sqrt(mu / p)
    r_P = np.array([r * math.cos(f), r * math.sin(f), 0.0])
    v_P = np.array([-speed_scale * math.sin(f), speed_scale * (e + math.cos(f)), 0.0])

    # Step 4: the DCM to N from P, written out entry by entry. Its columns
    #         are the perifocal unit vectors described in N.
    cO, sO = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(i), math.sin(i)
    C_NP = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])

    # Step 5: the same vectors described in N.
    return C_NP @ r_P, C_NP @ v_P


def rv2coe_single(r_N, v_N, mu, tol: float = 1e-10):
    """Classical elements from position and velocity, for one state.

    Parameters
    ----------
    r_N, v_N : array-like, shape (3,)
    mu : float
    tol : float
        Threshold for "circular" (e < tol) and "equatorial".

    Returns
    -------
    a, e, i, raan, argp, f : float
        Angles in radians; ``raan``, ``argp`` and ``f`` in [0, 2 pi).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, orbital elements and their singularities.
    """
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)
    r = np.linalg.norm(r_N)
    v_squared = v_N @ v_N

    # Step 1: angular momentum.
    h_N = np.cross(r_N, v_N)
    h = np.linalg.norm(h_N)
    h_hat_N = h_N / h

    # Step 2: eccentricity vector (points to the periapsis).
    e_N = ((v_squared - mu / r) * r_N - (r_N @ v_N) * v_N) / mu
    e = np.linalg.norm(e_N)

    # Step 3: semi-major axis from the energy.
    energy = 0.5 * v_squared - mu / r
    a = math.inf if energy == 0.0 else -mu / (2.0 * energy)

    # Step 4: inclination.
    i = math.acos(max(-1.0, min(1.0, h_hat_N[2])))

    # Step 5: node vector (points to the ascending node).
    node_N = np.cross([0.0, 0.0, 1.0], h_N)
    node = np.linalg.norm(node_N)

    # Step 6: is this state singular?
    equatorial = node < tol * h
    circular = e < tol

    def angle_in_plane(from_N, to_N):
        """Angle from one in-plane vector to another, positive along the motion."""
        return math.atan2(h_hat_N @ np.cross(from_N, to_N), from_N @ to_N)

    # Steps 7 and 8: the three angles, with the singular-case conventions.
    if equatorial:
        raan = 0.0
        node_hat_N = np.array([1.0, 0.0, 0.0])   # measure from axis 1 of N
    else:
        raan = math.atan2(node_N[1], node_N[0])
        node_hat_N = node_N / node

    if circular:
        argp = 0.0
        f = angle_in_plane(node_hat_N, r_N)      # no periapsis: measure from the node
    else:
        argp = angle_in_plane(node_hat_N, e_N)
        f = angle_in_plane(e_N, r_N)

    # Step 9: angles reduced to [0, 2 pi).
    return float(a), float(e), i, raan % TWO_PI, argp % TWO_PI, f % TWO_PI


def M2E_single(M, e, tol: float = 1e-13, max_iter: int = 50):
    """Eccentric anomaly from the mean anomaly, for one value (Newton's method).

    Solves Kepler's equation, E - e sin E = M, for E.

    Parameters
    ----------
    M : float
        Mean anomaly, radians (any value).
    e : float
        Eccentricity, 0 <= e < 1.
    tol : float
        Stop when the correction is below ``tol`` radians.
    max_iter : int

    Returns
    -------
    E : float

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 4, numerical solution and algorithmic aspects.
    """
    if not 0.0 <= e < 1.0:
        raise ValueError("Kepler's equation in this form holds for 0 <= e < 1.")

    # Step 1: remove the whole revolutions, so that M is in [-pi, pi].
    revolutions = round(M / TWO_PI)
    M_reduced = M - TWO_PI * revolutions

    # Step 2: starting value.
    if e < 0.8:
        E = M_reduced
    else:
        E = math.copysign(math.pi, M_reduced) if M_reduced != 0.0 else 0.0

    # Step 3: Newton's iteration on  g(E) = E - e sin E - M = 0,
    #         with  g'(E) = 1 - e cos E:   E <- E - g(E) / g'(E).
    for _ in range(max_iter):
        correction = (E - e * math.sin(E) - M_reduced) / (1.0 - e * math.cos(E))
        E = E - correction
        if abs(correction) < tol:
            break
    else:
        raise RuntimeError("Kepler's equation: no convergence within max_iter iterations.")

    # Step 4: put the whole revolutions back.
    return E + TWO_PI * revolutions


__all__ = ["coe2rv_single", "rv2coe_single", "M2E_single"]
