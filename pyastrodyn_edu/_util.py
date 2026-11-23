"""Small array helpers shared by the modules (not part of the public interface).

HOW TO READ THIS LIBRARY
------------------------
Every function accepts ONE state or MANY states in a single call:

    one state :  a vector has shape (3,)    and a scalar is a plain number
    n states  :  a vector has shape (n, 3)  and a scalar has shape (n,)
                 (one ROW per state)

The same lines of code serve both cases. The trick is always the same: the
three components of a vector live on the LAST axis, so the code says "last
axis" (``axis=-1``) instead of "axis 0". The helpers below hide that detail.

The file ``docs/numpy_primer.md`` explains, with small examples, the few NumPy
idioms used in the library. Comments in the code refer to it as "primer N".
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def dot(a, b):
    """Dot product a . b of two vectors, or of n pairs of vectors.

    One state:  (3,) . (3,)       -> a number
    n states :  (n, 3) . (n, 3)   -> shape (n,), one dot product per row
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Multiply component by component, then add up the three components,
    # which sit on the last axis (primer 2).
    return np.sum(a * b, axis=-1)


def norm(a):
    """Length |a| of a vector, or the lengths of n vectors (one per row)."""
    return np.linalg.norm(np.asarray(a, dtype=float), axis=-1)  # primer 2


def col(s):
    """Prepare a scalar so that it can multiply a vector.

    One state:  a number stays (in effect) a number.
    n states :  shape (n,) becomes shape (n, 1): a column. NumPy then
                multiplies row k of an (n, 3) array by the k-th scalar
                (primer 3). Without this step NumPy would refuse, or worse,
                pair the scalars with the three COMPONENTS.
    """
    return np.asarray(s, dtype=float)[..., np.newaxis]


def wrap_2pi(angle):
    """Reduce an angle to the interval [0, 2 pi)."""
    wrapped = np.mod(angle, TWO_PI)
    # For a tiny negative input, rounding can return exactly 2 pi; send it to 0.
    return np.where(wrapped >= TWO_PI, 0.0, wrapped)


def out(x):
    """Tidy a result: a plain NumPy number for one state, an array for many.

    Some NumPy functions return a "zero-dimensional array" for a single value.
    It behaves like a number but prints awkwardly; ``x[()]`` unwraps it.
    """
    x = np.asarray(x)
    return x[()] if x.ndim == 0 else x
