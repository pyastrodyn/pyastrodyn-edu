"""PyAstroDyn-edu: educational astrodynamics in Python.

Companion library of the graduate course in orbital mechanics of the
Technical University of Munich (winter semester 2026/27).

Every function is available from the top level, for example::

    import pyastrodyn_edu as ad
    r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, ad.MU_EARTH)

Copyright (c) 2026 Dr. Marcello Romano. MIT License; see the file LICENSE.
"""

from . import (anomalies, conics, constants, elements, kepler, lagrange,
               plotting, rotations, single_state, twobody, units)
from .anomalies import *  # noqa: F401,F403
from .conics import *  # noqa: F401,F403
from .constants import *  # noqa: F401,F403
from .elements import *  # noqa: F401,F403
from .kepler import *  # noqa: F401,F403
from .lagrange import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from .rotations import *  # noqa: F401,F403
from .single_state import *  # noqa: F401,F403
from .twobody import *  # noqa: F401,F403
from .units import *  # noqa: F401,F403

# --- Lecture 6: rocket equation and impulsive maneuvers ----------------------
from . import maneuvers  # noqa: E402
from .maneuvers import *  # noqa: F401,F403,E402

# --- Version -----------------------------------------------------------------
__version__ = "0.0.12"
