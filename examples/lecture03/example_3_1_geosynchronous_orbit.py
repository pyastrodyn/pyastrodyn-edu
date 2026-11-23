# %% [markdown]
# # Example problem 3.1: The geosynchronous orbit
#
# **Lecture 3, elliptical and circular orbits; the geosynchronous orbit.**
# Inspired by the worked examples of Curtis, *Orbital Mechanics for
# Engineering Students*; the statement and the numbers were written for this
# library.
#
# **Problem.** Find the radius, the altitude and the speed of the circular
# orbit whose period equals the sidereal rotation period of the Earth.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s.

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
#
# The sidereal day is the rotation period of the Earth with respect to the
# stars: about 4 minutes shorter than the 24 hours of the solar day.

# %%
sidereal_day = ad.SIDEREAL_DAY   # rotation period of the Earth, s (original: ad.SIDEREAL_DAY)
R = ad.R_EARTH                   # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH                 # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

print(f"sidereal day = {sidereal_day:.4f} s = {sidereal_day / 3600.0:.5f} h")

# %% [markdown]
# ## 2. Solution with the library

# %%
r = ad.synchronous_radius(sidereal_day, mu)
altitude = r - R
v = ad.circular_speed(r, mu)

print(f"radius   = {r:.3f} km")
print(f"altitude = {altitude:.3f} km")
print(f"speed    = {v:.6f} km/s")

# %% [markdown]
# ## 3. The same solution, step by step
#
# The period of a circular orbit of radius r is T = 2 pi sqrt(r^3 / mu).
# Solved for the radius: r = (mu T^2 / (4 pi^2))^(1/3). The speed on a
# circular orbit is sqrt(mu / r).

# %%
r_steps = (mu * sidereal_day**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
v_steps = np.sqrt(mu / r_steps)

assert np.isclose(r_steps, r) and np.isclose(v_steps, v)
print(f"r = {r_steps:.3f} km, v = {v_steps:.6f} km/s: sections 2 and 3 agree.")
print(f"the radius is {r / R:.3f} Earth radii")

# %% [markdown]
# ## 4. Independent check
#
# Two other routes: the speed must equal the circumference divided by the
# period, and the period computed from the radius must return the sidereal
# day.

# %%
v_check = 2.0 * np.pi * r / sidereal_day
T_check = ad.period(r, mu)

print(f"circumference / period = {v_check:.6f} km/s")
print(f"period from the radius = {T_check:.4f} s")

assert np.isclose(v_check, v) and np.isclose(T_check, sidereal_day)
print("Both checks agree.")

# %% [markdown]
# ## 5. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"sidereal_day": sidereal_day, "R": R, "mu": mu}
results = {"r": r, "altitude": altitude, "v": v}
validation.compare("L3-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Use the solar day, `sidereal_day = 86400.0`. By how many km does the
#    radius change? Why is the sidereal day the right one?
# 2. Mars: `mu = 42828.37`, `R = 3396.2`, rotation period `88642.66` s. Find
#    the altitude of the synchronous orbit of Mars.
# 3. A satellite must pass over the same place twice per sidereal day. What
#    is the radius of its circular orbit?
