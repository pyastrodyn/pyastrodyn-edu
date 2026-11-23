# %% [markdown]
# # Example problem 3.2: A low Earth orbit in canonical units
#
# **Lecture 3, scaling of the equations.** Inspired by Vallado,
# *Fundamentals of Astrodynamics and Applications*; the statement and the
# numbers were written for this library.
#
# **Problem.** A spacecraft is on a circular orbit at an altitude of 400 km.
# Find its period. Then take the orbit radius as the unit of length and
# choose the unit of time so that the gravitational parameter equals 1: find
# the unit of time, and the period, the speed and the gravitational parameter
# in these units.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units of the data: km and s. In the canonical system the unit of length is
# called DU and the unit of time TU.

# %%
import numpy as np

import pyastrodyn_edu as ad
from pyastrodyn_edu import validation

# %% [markdown]
# ## 1. Given data
#
# These are the numbers to play with. To return to the original problem,
# type the original values back (they are given in the comments) and run
# everything again.

# %%
altitude = 400.0     # altitude of the circular orbit, km (original: 400.0)
R = ad.R_EARTH       # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The orbit in km and s

# %%
r = R + altitude
period = ad.period(r, mu)
speed = ad.circular_speed(r, mu)

print(f"radius = {r:.3f} km")
print(f"period = {period:.3f} s = {period / 60.0:.2f} min")
print(f"speed  = {speed:.6f} km/s")

# %% [markdown]
# ## 3. The canonical unit system
#
# Choose the unit of length DU = r. The gravitational parameter has
# dimensions length^3 / time^2, so it equals 1 in the new units if the unit
# of time is TU = sqrt(DU^3 / mu). `UnitSystem.canonical` builds this system.

# %%
canonical = ad.UnitSystem.canonical(mu, r)

time_unit_steps = np.sqrt(r**3 / mu)         # the same, written out
assert np.isclose(canonical.time, time_unit_steps) and canonical.length == r

print(f"DU = {canonical.length:.3f} km")
print(f"TU = {canonical.time:.6f} s")
print(f"unit of speed DU/TU = {canonical.unit('speed'):.6f} km/s")

# %% [markdown]
# ## 4. The orbit in canonical units
#
# `nondim` converts a value from km and s into the canonical system; the
# second argument says what kind of quantity it is.

# %%
period_canonical = canonical.nondim(period, "time")
speed_canonical = canonical.nondim(speed, "speed")
mu_canonical = canonical.nondim(mu, "mu")

print(f"period = {period_canonical:.12f} TU   (2 pi = {2.0 * np.pi:.12f})")
print(f"speed  = {speed_canonical:.12f} DU/TU")
print(f"mu     = {mu_canonical:.12f} DU^3/TU^2")

# %% [markdown]
# ## 5. Independent check: compute directly in canonical units
#
# The functions of the library have no unit of their own: they are correct
# in any consistent system. So the same orbit can be computed with radius 1
# and mu = 1, and the results converted back with `dim`. In these units the
# period of the reference orbit is exactly 2 pi and its speed exactly 1.

# %%
period_direct = ad.period(1.0, 1.0)                    # radius 1 DU, mu = 1
speed_direct = ad.circular_speed(1.0, 1.0)

assert np.isclose(period_direct, 2.0 * np.pi) and np.isclose(speed_direct, 1.0)
assert np.isclose(period_canonical, period_direct) and np.isclose(speed_canonical, speed_direct)
assert np.isclose(canonical.dim(period_direct, "time"), period)
assert np.isclose(canonical.dim(speed_direct, "speed"), speed)
print(f"computed in canonical units: period {period_direct:.6f} TU, speed {speed_direct:.1f} DU/TU")
print(f"converted back: period {canonical.dim(period_direct, 'time'):.3f} s, "
      f"speed {canonical.dim(speed_direct, 'speed'):.6f} km/s")
print("The two unit systems give the same orbit.")

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"altitude": altitude, "R": R, "mu": mu}
results = {"period": period, "time_unit": canonical.time, "period_canonical": period_canonical,
           "speed_canonical": speed_canonical, "mu_canonical": mu_canonical}
validation.compare("L3-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Change the altitude to 20200 km (a navigation satellite). Which results
#    of section 4 change?
# 2. In the canonical system of this orbit, what is the period of a circular
#    orbit of radius 4 DU? Compute it with `ad.period(4.0, 1.0)` and explain
#    the number with Kepler's third law.
# 3. Why is a scaled system useful in numerical work? See section 6 of the
#    Lecture 1 notebook.
