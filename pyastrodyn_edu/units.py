"""Scaling of the equations: unit systems and canonical units.

The functions of the library carry no unit of their own: they are correct in
any consistent set of length unit, time unit and ``mu``. This module provides
the conversion between two such sets, in one place.

Rule of use: a script chooses its unit system once, converts its inputs at the
start and its outputs at the end, and never mixes systems in between.
"""

import math
from dataclasses import dataclass

import numpy as np

_KINDS = ("length", "time", "speed", "acceleration", "mu")


@dataclass(frozen=True)
class UnitSystem:
    """A consistent pair of length unit and time unit.

    Parameters
    ----------
    length : float
        The length unit, expressed in the base length unit (km in this library's
        constants).
    time : float
        The time unit, expressed in the base time unit (s).

    Examples
    --------
    Canonical units of a low Earth orbit, in which ``mu = 1``:

    >>> from pyastrodyn_edu.constants import MU_EARTH
    >>> canonical = UnitSystem.canonical(MU_EARTH, 6778.0)
    >>> float(round(canonical.nondim(MU_EARTH, "mu"), 12))
    1.0

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.
    Lecture 3, scaling of the equations.
    """

    length: float
    time: float

    @classmethod
    def canonical(cls, mu: float, r_ref: float) -> "UnitSystem":
        """Canonical units: length unit ``r_ref``, time unit such that ``mu = 1``.

        The time unit is ``sqrt(r_ref**3 / mu)``. A circular orbit of radius
        ``r_ref`` then has speed 1 and period 2 pi.
        """
        return cls(length=float(r_ref), time=math.sqrt(r_ref**3 / mu))

    def unit(self, kind: str) -> float:
        """Size of the unit of the given kind, in base units.

        ``kind`` is one of "length", "time", "speed", "acceleration", "mu".
        """
        if kind == "length":
            return self.length
        if kind == "time":
            return self.time
        if kind == "speed":
            return self.length / self.time
        if kind == "acceleration":
            return self.length / self.time**2
        if kind == "mu":
            return self.length**3 / self.time**2
        raise ValueError(f"kind must be one of {_KINDS}, not {kind!r}")

    def nondim(self, value, kind: str):
        """Convert a value from base units into this unit system."""
        return np.asarray(value, dtype=float)[()] / self.unit(kind)

    def dim(self, value, kind: str):
        """Convert a value from this unit system back into base units."""
        return np.asarray(value, dtype=float)[()] * self.unit(kind)


__all__ = ["UnitSystem"]
