# %% [markdown]
# # Example problem 4.3: The three anomalies along an elliptical orbit
#
# **Lecture 4, mean anomaly and Kepler's equation.** Inspired by the worked
# examples of Curtis, *Orbital Mechanics for Engineering Students*; the
# statement and the numbers were written for this library.
#
# **Problem.** On an orbit with a semi-major axis of 15000 km and an
# eccentricity of 0.5, plot the mean anomaly M, the eccentric anomaly E and
# the true anomaly f against the time, over one period. Find the three
# anomalies a quarter of a period after the perigee. Find the largest
# difference between the true and the mean anomaly, and where it occurs.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The angles are computed in radians and printed and plotted
# in degrees.

# %%
import matplotlib.pyplot as plt
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
a = 15000.0          # semi-major axis, km (original: 15000.0)
e = 0.5              # eccentricity, between 0 and 1 (original: 0.5)
fraction = 0.25      # fraction of the period after the perigee (original: 0.25)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# The mean anomaly grows uniformly with the time: M = n t, with t measured
# from the perigee. `M2E` solves Kepler's equation for the eccentric anomaly,
# and `E2f` gives the true anomaly. Both accept many values in one call.

# %%
T = ad.period(a, mu)
n = ad.mean_motion(a, mu)                    # mean motion, rad/s
t = np.linspace(0.0, T, 721)                 # one period, from perigee to perigee
M = n * t                                    # mean anomaly at each time
E = ad.M2E(M, e)                             # eccentric anomaly (Kepler's equation)
f = np.unwrap(ad.E2f(E, e))                  # true anomaly, growing from 0 to 2 pi

M_q = 2.0 * np.pi * fraction                 # the three anomalies at the given fraction of the period
E_q = ad.M2E(M_q, e)
f_q = ad.E2f(E_q, e)
print(f"period = {T:.1f} s = {T / 3600.0:.3f} h")
print(f"at {fraction:g} of the period: M = {np.degrees(M_q):.4f} deg, E = {np.degrees(E_q):.4f} deg, "
      f"f = {np.degrees(f_q):.4f} deg")

# %% [markdown]
# The figure. On the left the three anomalies against the time. On the right
# the differences from the mean anomaly: they vanish at the perigee and at
# the apogee, where the three anomalies coincide.

# %%
hours = t / 3600.0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
ax1.plot(hours, np.degrees(M), label="mean anomaly M")
ax1.plot(hours, np.degrees(E), label="eccentric anomaly E")
ax1.plot(hours, np.degrees(f), label="true anomaly f")
ax1.plot(fraction * T / 3600.0 * np.ones(3), np.degrees([M_q, E_q, f_q]), "ko", ms=4)
ax1.set_xlabel("time from the perigee, h")
ax1.set_ylabel("anomaly, deg")
ax1.set_yticks(range(0, 361, 90))
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(hours, np.degrees(E - M), color="tab:orange", label="E - M")
ax2.plot(hours, np.degrees(f - M), color="tab:green", label="f - M")
ax2.axhline(0.0, color="0.5", lw=0.8)
ax2.set_xlabel("time from the perigee, h")
ax2.set_ylabel("difference, deg")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The largest difference between the true and the mean anomaly
#
# The difference f - M is called the equation of the centre. It is largest
# where f and M grow at the same rate. From the conservation of the angular
# momentum, df/dM = (a / r)^2 sqrt(1 - e^2): the two rates are equal where
# r = a (1 - e^2)^(1/4). The orbit equation gives f there, and `f2M` gives M.

# %%
r_largest = a * (1.0 - e**2) ** 0.25
f_largest = np.arccos((ad.semilatus_rectum(a, e) / r_largest - 1.0) / e)     # in the first half of the orbit
M_largest = ad.f2M(f_largest, e)
largest_difference = f_largest - M_largest
time_at_largest = M_largest / n
print(f"largest f - M = {np.degrees(largest_difference):.4f} deg, at f = {np.degrees(f_largest):.4f} deg, "
      f"M = {np.degrees(M_largest):.4f} deg")
print(f"it occurs {time_at_largest:.1f} s after the perigee ({time_at_largest / T:.3f} of the period), "
      f"at the distance {r_largest:.1f} km")

# %% [markdown]
# ## 4. Independent checks
#
# 1. The largest value of f - M on the grid of section 2 agrees with section 3.
# 2. Kepler's equation and the half-angle relation hold for the three
#    anomalies of section 2.
# 3. Before the apogee M < E < f: the spacecraft is ahead of the uniform
#    motion. After the apogee the order is reversed.

# %%
k = np.argmax(f - M)
print(f"on the grid: largest f - M = {np.degrees(f[k] - M[k]):.4f} deg at t = {t[k]:.1f} s")
assert np.isclose(f[k] - M[k], largest_difference, rtol=1e-4)
assert np.isclose(E_q - e * np.sin(E_q), M_q)
assert np.isclose(np.tan(f_q / 2.0), np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E_q / 2.0))
first_half = (t > 0.0) & (t < 0.5 * T)
second_half = (t > 0.5 * T) & (t < T)
assert np.all(M[first_half] < E[first_half]) and np.all(E[first_half] < f[first_half])
assert np.all(M[second_half] > E[second_half]) and np.all(E[second_half] > f[second_half])
print("The checks agree.")

# %% [markdown]
# ## 5. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "fraction": fraction, "mu": mu}
results = {"period": T, "M": np.degrees(M_q), "E": np.degrees(E_q), "f": np.degrees(f_q),
           "largest_difference": np.degrees(largest_difference), "f_at_largest": np.degrees(f_largest),
           "M_at_largest": np.degrees(M_largest), "time_at_largest": time_at_largest}
validation.compare("L4-3", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `e = 0.1`, then `e = 0.9`. How does the largest difference change?
#    For a small eccentricity it is close to 2 e radians.
# 2. Set `e = 0.0`. What happens to the three curves?
# 3. The semi-major axis changes the period, not the shape of the curves
#    against the fraction of the period: verify it with `a = 42164.0`.
