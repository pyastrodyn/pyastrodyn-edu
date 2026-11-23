"""Conversion between the state (position, velocity) and orbital elements.

Classical elements, always in this order: ``a, e, i, raan, argp, f``
(semi-major axis, eccentricity, inclination, right ascension of the ascending
node, argument of periapsis, true anomaly). Angles in radians.

Every element may be one value or an array of n values; vectors have shape
(3,) or (n, 3), components in the inertial basis N.

FIRST READING: the module ``single_state`` contains ``coe2rv_single`` and
``rv2coe_single``, the same algorithms for ONE state, written with ordinary
``if`` statements. Read those first; the functions here do the same steps in
the same order, for many states at once. The tests check that they agree.
"""

import numpy as np

from ._util import col, dot, norm, out, wrap_2pi
from .rotations import change_basis, dcm_NP


def coe2rv(a, e, i, raan, argp, f, mu):
    """Position and velocity from the classical orbital elements.

    Parameters
    ----------
    a, e, i, raan, argp, f : float or ndarray, shape (n,)
        Classical elements. ``a`` is negative for a hyperbola.
    mu : float
        Gravitational parameter.

    Returns
    -------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
        Components in the inertial basis N.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, conversion between the state and the orbital elements.

    Notes
    -----
    Valid for the ellipse and the hyperbola; not for the parabola (``a``
    infinite). Closed-form: accurate to floating-point rounding. For the
    singular cases see ``rv2coe``: the two functions use the same
    conventions, so they are exact inverses of each other there too.
    """
    # Step 0: give all inputs a common shape (primer 1). One orbit combined
    #         with n true anomalies becomes n full sets of elements.
    #         Shapes after this line: all () for one state, all (n,) for n.
    a, e, i, raan, argp, f, mu = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (a, e, i, raan, argp, f, mu))
    )

    # Step 1: semi-latus rectum,  p = a (1 - e^2).
    #         Positive for the ellipse (a>0, e<1) and the hyperbola (a<0, e>1).
    p = a * (1.0 - e**2)
    if np.any(p <= 0.0):
        raise ValueError("a (1 - e^2) must be positive: ellipse (a>0, e<1) or hyperbola (a<0, e>1).")

    # Step 2: distance from the orbit equation,  r = p / (1 + e cos f).
    cos_f, sin_f = np.cos(f), np.sin(f)
    r = p / (1.0 + e * cos_f)                    # shape () or (n,)

    # Step 3: position and velocity in the PERIFOCAL basis P (axis 1 to the
    #         periapsis, axis 3 along h). The motion is in the plane 1-2:
    #             r_P = r [ cos f,  sin f,  0 ]
    #             v_P = sqrt(mu/p) [ -sin f,  e + cos f,  0 ]
    #         "np.stack(..., axis=-1)" puts the three components side by
    #         side on the last axis (primer 4): shape (3,) or (n, 3).
    speed_scale = np.sqrt(mu / p)                # = mu / h
    zero = np.zeros_like(r)
    r_P = np.stack((r * cos_f, r * sin_f, zero), axis=-1)
    v_P = np.stack((-speed_scale * sin_f, speed_scale * (e + cos_f), zero), axis=-1)

    # Step 4: the same two vectors, described in the inertial basis N.
    C_NP = dcm_NP(i, raan, argp)                 # to N, from P; (3,3) or (n,3,3)
    r_N = change_basis(C_NP, r_P)
    v_N = change_basis(C_NP, v_P)
    return r_N, v_N


def rv2coe(r_N, v_N, mu, tol: float = 1e-10):
    """Classical orbital elements from position and velocity.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
        Position and velocity, components in the inertial basis N.
    mu : float
        Gravitational parameter.
    tol : float
        An orbit is treated as circular if e < tol, and as equatorial if the
        node vector is shorter than tol times the angular momentum.

    Returns
    -------
    a, e, i, raan, argp, f : float or ndarray, shape (n,)
        Angles in radians; ``raan``, ``argp`` and ``f`` in [0, 2 pi).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, orbital elements and their singularities.

    Notes
    -----
    Singular cases, selected state by state (a mixed set of states is
    handled in one call):

    - circular, inclined: ``argp = 0`` and ``f`` is the argument of latitude
      (angle from the ascending node to the position);
    - elliptical, equatorial: ``raan = 0`` and ``argp`` is the longitude of
      periapsis (angle from axis 1 of N to the periapsis, measured in the
      direction of motion);
    - circular, equatorial: ``raan = 0``, ``argp = 0`` and ``f`` is the true
      longitude (angle from axis 1 of N to the position).

    For the parabola the returned ``a`` is infinite. Closed-form: accurate
    to floating-point rounding, except that close to a singular case (e or
    the node vector comparable with ``tol``) the angles ``raan``, ``argp``
    and ``f`` lose accuracy individually, while their sum does not.
    """
    # Step 0: common shape for position and velocity (primer 1).
    r_N, v_N = np.broadcast_arrays(np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float))
    mu = np.asarray(mu, dtype=float)
    r = norm(r_N)                                # distance, shape () or (n,)
    v_squared = dot(v_N, v_N)                    # speed squared

    # Step 1: angular momentum,  h = r x v.  Normal to the orbital plane.
    h_N = np.cross(r_N, v_N)                     # shape (3,) or (n, 3)
    h = norm(h_N)
    h_hat_N = h_N / col(h)                       # unit vector along h

    # Step 2: eccentricity vector,  e = ((v^2 - mu/r) r - (r . v) v) / mu.
    #         It points to the periapsis and its length is the eccentricity.
    #         "col(...)" lets a scalar multiply a vector for every state
    #         (primer 3).
    e_N = (col(v_squared - mu / r) * r_N - col(dot(r_N, v_N)) * v_N) / col(mu)
    e = norm(e_N)

    # Step 3: semi-major axis from the energy,  energy = v^2/2 - mu/r = -mu/(2a).
    energy = 0.5 * v_squared - mu / r
    with np.errstate(divide="ignore"):           # parabola: energy = 0, a = inf
        a = -mu / (2.0 * energy)

    # Step 4: inclination = angle between h and axis 3 of N,  cos i = h_3 / h.
    #         "[..., 2]" is the third component of every vector (primer 4).
    #         The clip guards against 1.0000000000000002 from rounding.
    i = np.arccos(np.clip(h_hat_N[..., 2], -1.0, 1.0))

    # Step 5: node vector,  n = (axis 3 of N) x h.  It lies along the line of
    #         nodes and points to the ASCENDING node. For an equatorial orbit
    #         h is parallel to axis 3 and n vanishes: the node is undefined.
    axis1_N = np.array([1.0, 0.0, 0.0])
    axis3_N = np.array([0.0, 0.0, 1.0])
    node_N = np.cross(axis3_N, h_N)
    node = norm(node_N)

    # Step 6: which states are singular? Two yes/no flags PER STATE
    #         (a single True/False for one state, an array of n for many).
    equatorial = node < tol * h
    circular = e < tol

    # Step 7: the direction in the orbital plane from which argp is measured.
    #         Regular case: the ascending node. Equatorial case: axis 1 of N.
    #         "np.where(flag, x, y)" picks x where the flag is True and y
    #         elsewhere, state by state: it is the many-states form of
    #         "if flag: x else: y" (primer 6). The inner np.where only avoids
    #         a division by zero in the states that are then discarded.
    safe_node = np.where(equatorial, 1.0, node)
    node_hat_N = np.where(col(equatorial), axis1_N, node_N / col(safe_node))

    def angle_in_plane(from_N, to_N):
        """Angle from one in-plane vector to another, positive along the motion.

        For two vectors u, w in the orbital plane:  sin = h_hat . (u x w),
        cos = u . w  (both times |u||w|, which cancels in atan2). Using
        atan2 gives the right quadrant with no further test.
        """
        return np.arctan2(dot(h_hat_N, np.cross(from_N, to_N)), dot(from_N, to_N))

    # Step 8: the three angles, each with its singular-case convention.
    #   raan: angle of the node in the plane 1-2 of N; 0 if equatorial.
    raan = np.where(equatorial, 0.0, np.arctan2(node_N[..., 1], node_N[..., 0]))
    #   argp: from the node (or axis 1) to the periapsis; 0 if circular.
    argp = np.where(circular, 0.0, angle_in_plane(node_hat_N, e_N))
    #   f: from the periapsis to the position; if circular there is no
    #      periapsis, so it is measured from the node (or axis 1) instead.
    reference_N = np.where(col(circular), node_hat_N, e_N)
    f = angle_in_plane(reference_N, r_N)

    # Step 9: angles reduced to [0, 2 pi); tidy the outputs.
    return (out(a), out(e), out(i), out(wrap_2pi(raan)), out(wrap_2pi(argp)), out(wrap_2pi(f)))


def coe2equinoctial(a, e, i, raan, argp, f):
    """Modified equinoctial elements from the classical elements.

    Returns
    -------
    p, ex, ey, hx, hy, L : float or ndarray, shape (n,)
        ``p = a (1 - e^2)``; ``ex = e cos(raan + argp)``,
        ``ey = e sin(raan + argp)``; ``hx = tan(i/2) cos(raan)``,
        ``hy = tan(i/2) sin(raan)``; true longitude ``L = raan + argp + f``,
        in [0, 2 pi).

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, orbital elements (classical, their singularities, equinoctial).

    Notes
    -----
    This is the MODIFIED equinoctial set. It differs from the classical
    (direct) equinoctial set in two elements: the first is the semi-latus
    rectum p in place of the semi-major axis a, and the last is the TRUE
    longitude L in place of the mean longitude. The four middle elements are
    the same quantities in both sets. The modified set is regular for
    circular and for equatorial orbits, and also valid for the hyperbola; it
    is singular only for i = pi (retrograde equatorial).
    """
    a, e, i, raan, argp, f = (np.asarray(x, dtype=float) for x in (a, e, i, raan, argp, f))

    # Step 1: semi-latus rectum.
    p = a * (1.0 - e**2)

    # Step 2: the eccentricity vector, by its two components in the
    #         "equinoctial" basis: (ex, ey) = e (cos, sin) of the longitude of
    #         periapsis, raan + argp. For e -> 0 both go smoothly to zero.
    longitude_of_periapsis = raan + argp
    ex = e * np.cos(longitude_of_periapsis)
    ey = e * np.sin(longitude_of_periapsis)

    # Step 3: the tilt of the orbital plane: (hx, hy) = tan(i/2) (cos, sin)
    #         of raan. For i -> 0 both go smoothly to zero.
    tan_half_i = np.tan(0.5 * i)
    hx = tan_half_i * np.cos(raan)
    hy = tan_half_i * np.sin(raan)

    # Step 4: true longitude, the sum of the three angles; always defined.
    L = wrap_2pi(longitude_of_periapsis + f)
    return out(p), out(ex), out(ey), out(hx), out(hy), out(L)


def equinoctial2coe(p, ex, ey, hx, hy, L):
    """Classical elements from the modified equinoctial elements.

    Returns
    -------
    a, e, i, raan, argp, f : float or ndarray, shape (n,)

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, orbital elements (classical, their singularities, equinoctial).

    Notes
    -----
    Where the classical set is singular the split of the true longitude into
    ``raan``, ``argp`` and ``f`` is a convention: ``raan = 0`` for an
    equatorial orbit and ``argp = -raan`` (modulo 2 pi) for a circular one.
    The state obtained with ``coe2rv`` is correct in every case.
    """
    p, ex, ey, hx, hy, L = (np.asarray(x, dtype=float) for x in (p, ex, ey, hx, hy, L))

    # Step 1: lengths of the two small vectors give e and i.
    e = np.hypot(ex, ey)                         # sqrt(ex^2 + ey^2)
    i = 2.0 * np.arctan(np.hypot(hx, hy))        # tan(i/2) = sqrt(hx^2 + hy^2)

    # Step 2: their directions give raan and the longitude of periapsis.
    #         (atan2(0, 0) is 0 in NumPy: that sets the singular-case
    #         conventions stated in the Notes.)
    raan = np.arctan2(hy, hx)
    longitude_of_periapsis = np.arctan2(ey, ex)

    # Step 3: back to a, and split the longitudes into the classical angles.
    a = p / (1.0 - e**2)
    argp = longitude_of_periapsis - raan
    f = L - longitude_of_periapsis
    return out(a), out(e), out(i), out(wrap_2pi(raan)), out(wrap_2pi(argp)), out(wrap_2pi(f))


__all__ = ["coe2rv", "rv2coe", "coe2equinoctial", "equinoctial2coe"]
