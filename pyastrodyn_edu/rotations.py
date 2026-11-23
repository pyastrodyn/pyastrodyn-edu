"""Direction cosine matrices (DCMs).

THE CONVENTION (passive), fixed for the whole library
-----------------------------------------------------
``C_BA`` reads "to B, from A". It takes the components of a vector in basis A
and returns the components of THE SAME vector in basis B:

    v_B = C_BA @ v_A

Nothing rotates. The vector stays where it is; only its description changes,
because we look at it from another basis. That is why this module has a
function called ``change_basis`` and none called "rotate".

A BASIS (in the slides of the course: vector base) is a set of three
mutually orthogonal unit vectors, fixed in a frame. It is not a frame (the
reference with respect to which motion is measured) and not a Cartesian
coordinate system (an origin together with a basis). See the conventions in
the README.

Basis letters: N inertial, P perifocal, H Hill (axis 1 radial, axis 2
transverse, axis 3 along the orbital angular momentum), E Earth-fixed.

SHAPES
------
An angle may be one value or an array of n values. A DCM is then (3, 3) for
one state or (n, 3, 3) for n states: a stack of n matrices.

- One state: plain matrix algebra works, ``v_B = C_BA @ v_A``.
- Many states: use ``change_basis(C_BA, v_A)``, which pairs matrix k with
  vector k.
"""

import numpy as np

from ._util import col, norm


def _elementary(angle, axis: int) -> np.ndarray:
    """Build the elementary DCM about one axis (0, 1 or 2 in Python counting).

    For a turn of the BASIS by ``angle`` about its axis 3, the matrix is

        [  cos  sin   0 ]
        [ -sin  cos   0 ]        (note where the minus sign is: passive)
        [   0    0    1 ]

    and the other two axes follow by cyclic permutation.
    """
    angle = np.asarray(angle, dtype=float)       # shape () or (n,)
    c, s = np.cos(angle), np.sin(angle)          # same shape as angle

    # Step 1: an array of zeros with room for one 3x3 matrix per angle.
    #         angle.shape + (3, 3) is (3, 3) for one angle, (n, 3, 3) for n.
    C = np.zeros(angle.shape + (3, 3))

    # Step 2: the two axes that turn. Cyclic: about axis 0 -> (1, 2),
    #         about axis 1 -> (2, 0), about axis 2 -> (0, 1).
    j, k = [(1, 2), (2, 0), (0, 1)][axis]

    # Step 3: fill the entries. "C[..., j, k]" means entry (j, k) of EVERY
    #         matrix in the stack (primer 4).
    C[..., axis, axis] = 1.0     # the rotation axis itself is unchanged
    C[..., j, j] = c
    C[..., k, k] = c
    C[..., j, k] = s
    C[..., k, j] = -s
    return C


def rot1(angle) -> np.ndarray:
    """Elementary DCM for a turn of the basis about its axis 1.

    If basis B is obtained by turning basis A about its first axis by
    ``angle`` (radians, right-hand rule), then ``rot1(angle)`` is ``C_BA``.

    Parameters
    ----------
    angle : float or ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (3, 3) or (n, 3, 3)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, the orbit in three dimensions.
    """
    return _elementary(angle, 0)


def rot2(angle) -> np.ndarray:
    """Elementary DCM for a turn of the basis about its axis 2.

    See ``rot1`` for the meaning of the result and for the shapes.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, the orbit in three dimensions.
    """
    return _elementary(angle, 1)


def rot3(angle) -> np.ndarray:
    """Elementary DCM for a turn of the basis about its axis 3.

    See ``rot1`` for the meaning of the result and for the shapes.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, the orbit in three dimensions.
    """
    return _elementary(angle, 2)


def change_basis(C_BA, v_A) -> np.ndarray:
    """Components in basis B of a vector given by its components in basis A.

    ``v_B = C_BA @ v_A``: the same vector, described in another basis.

    For one DCM and one vector this is exactly ``C_BA @ v_A``. The function
    exists for the many-states cases, which the ``@`` operator does not cover:

    - one DCM, n vectors:  every vector is re-expressed with the same matrix;
    - n DCMs, n vectors:   vector k is re-expressed with matrix k.

    Parameters
    ----------
    C_BA : ndarray, shape (3, 3) or (n, 3, 3)
    v_A : ndarray, shape (3,) or (n, 3)

    Returns
    -------
    v_B : ndarray, shape (3,) or (n, 3)

    References
    ----------
    Schaub and Junkins, *Analytical Mechanics of Space Systems*.
    Lecture 5, the orbit in three dimensions.
    """
    C_BA = np.asarray(C_BA, dtype=float)
    v_A = np.asarray(v_A, dtype=float)
    # Written out, component i of the result is  v_B[i] = sum over j of
    # C_BA[i, j] * v_A[j].  The string below says exactly that; the "..."
    # stands for "the same, for every state in the stack" (primer 5).
    return np.einsum("...ij,...j->...i", C_BA, v_A)


def invert_dcm(C_BA) -> np.ndarray:
    """The DCM in the opposite direction: ``C_AB`` from ``C_BA``.

    For a DCM the inverse is the transpose (rows and columns exchanged), so
    no matrix inversion is needed.

    Parameters
    ----------
    C_BA : ndarray, shape (3, 3) or (n, 3, 3)

    Returns
    -------
    C_AB : ndarray, same shape

    References
    ----------
    Schaub and Junkins, *Analytical Mechanics of Space Systems*.
    Lecture 5, the orbit in three dimensions.
    """
    # Exchange the last two axes: rows <-> columns of every matrix in the
    # stack. (For one matrix this is just C_BA.T.)
    return np.swapaxes(np.asarray(C_BA, dtype=float), -1, -2)


def dcm_NP(i, raan, argp) -> np.ndarray:
    """DCM to the inertial basis N from the perifocal basis P.

    The perifocal basis is reached from N by the 3-1-3 sequence: ``raan``
    about axis 3, ``i`` about axis 1, ``argp`` about axis 3.

    Parameters
    ----------
    i, raan, argp : float or ndarray, shape (n,)
        Inclination, right ascension of the ascending node, argument of
        periapsis, in radians.

    Returns
    -------
    C_NP : ndarray, shape (3, 3) or (n, 3, 3)
        ``v_N = C_NP @ v_P``.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, conversion between the state and the orbital elements.
    """
    # Step 0: give the three angles a common shape, so that one value can be
    #         combined with many (primer 1).
    i, raan, argp = np.broadcast_arrays(
        np.asarray(i, dtype=float), np.asarray(raan, dtype=float), np.asarray(argp, dtype=float)
    )

    # Step 1: go from N to P with three successive turns of the basis.
    #         Read the product from RIGHT to LEFT: first raan about axis 3,
    #         then i about (the new) axis 1, then argp about (the new) axis 3.
    C_PN = rot3(argp) @ rot1(i) @ rot3(raan)     # shape (3, 3) or (n, 3, 3)

    # Step 2: we want the opposite direction, to N from P.
    return invert_dcm(C_PN)


def dcm_EN(theta_g) -> np.ndarray:
    """DCM to the Earth-fixed basis E from the inertial basis N.

    Simple model: the Earth-fixed basis is the inertial one turned about the
    common axis 3 (the rotation axis) by the Greenwich angle ``theta_g``.
    Precession, nutation and polar motion are not included.

    Parameters
    ----------
    theta_g : float or ndarray, shape (n,)
        Greenwich angle, radians.

    Returns
    -------
    C_EN : ndarray, shape (3, 3) or (n, 3, 3)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 5, ECI and ECEF coordinate systems.
    """
    return rot3(theta_g)


def dcm_HN(r_N, v_N) -> np.ndarray:
    """DCM to the Hill basis H from the inertial basis N.

    Hill basis: axis 1 radial (along the position), axis 3 along the orbital
    angular momentum, axis 2 completes the right-handed triad (transverse; it
    is the direction of the velocity only on a circular orbit and at the
    apses).

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
        Position and velocity, components in N.

    Returns
    -------
    C_HN : ndarray, shape (3, 3) or (n, 3, 3)
        ``v_H = C_HN @ v_N``. Its rows are the Hill unit vectors in N.

    References
    ----------
    Schaub and Junkins, *Analytical Mechanics of Space Systems*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, Hill vector base.

    Notes
    -----
    Not defined for rectilinear motion (position parallel to velocity).
    """
    r_N, v_N = np.broadcast_arrays(np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float))

    # Step 1: angular momentum, h = r x v. It is normal to the orbital plane.
    h_N = np.cross(r_N, v_N)                     # shape (3,) or (n, 3)
    h = norm(h_N)                                # shape () or (n,)
    if np.any(h == 0.0):
        raise ValueError("Hill basis undefined: position and velocity are parallel.")

    # Step 2: the three unit vectors of H, each by its components in N.
    radial_N = r_N / col(norm(r_N))              # axis 1: along r
    normal_N = h_N / col(h)                      # axis 3: along h
    transverse_N = np.cross(normal_N, radial_N)  # axis 2 = axis 3 x axis 1

    # Step 3: a DCM "to H, from N" has the unit vectors of H as its ROWS.
    #         "axis=-2" stacks them as rows of each matrix (primer 4).
    return np.stack((radial_N, transverse_N, normal_N), axis=-2)


__all__ = [
    "rot1", "rot2", "rot3", "change_basis", "invert_dcm", "dcm_NP", "dcm_EN", "dcm_HN",
]
