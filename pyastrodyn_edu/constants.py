"""Physical constants, each with its source. Units: km, s, kg.

No function of the library reads these values by itself: ``mu`` and every
other constant is always passed explicitly, so any other set of values, or a
scaled unit system, can be used.
"""

import math

# Gravitational parameter of the Earth, km^3/s^2.
# Source: WGS 84 (value that includes the mass of the atmosphere).
MU_EARTH = 398600.4418

# Equatorial radius of the Earth, km. Source: WGS 84.
R_EARTH = 6378.137

# Rotation rate of the Earth, rad/s. Source: WGS 84.
OMEGA_EARTH = 7.292115e-5

# Sidereal rotation period of the Earth, s, derived from OMEGA_EARTH.
SIDEREAL_DAY = 2.0 * math.pi / OMEGA_EARTH

# Standard acceleration of gravity, km/s^2. Exact by definition (CGPM, 1901).
# Used only to convert a specific impulse in seconds into an exhaust velocity.
G0 = 9.80665e-3

__all__ = ["MU_EARTH", "R_EARTH", "OMEGA_EARTH", "SIDEREAL_DAY", "G0"]
