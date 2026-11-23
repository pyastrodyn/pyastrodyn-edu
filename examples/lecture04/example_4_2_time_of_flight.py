# %% [markdown]
# # Example problem 4.2: Time of flight between two points of an orbit
#
# **Lecture 4, position as a function of time.** Inspired by the worked
# examples of Curtis, *Orbital Mechanics for Engineering Students*; the
# statement and the numbers were written for this library.
#
# **Problem.** On an orbit with a semi-major axis of 24400 km and an
# eccentricity of 0.73, how long does the spacecraft take to go from a true
# anomaly of 90 degrees to a true anomaly of 180 degrees?
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The true anomalies are given in degrees and converted to
# radians.

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
a = 24400.0          # semi-major axis, km (original: 24400.0)
e = 0.73             # eccentricity (original: 0.73)
f1_deg = 90.0        # true anomaly of the first point, degrees (original: 90.0)
f2_deg = 180.0       # true anomaly of the second point, degrees (original: 180.0)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

f1, f2 = np.radians(f1_deg), np.radians(f2_deg)             # the same angles in radians

# %% [markdown]
# ## 2. Solution with the library
#
# This is the inverse of Example problem 4.1: there the time was given and
# the position asked; here the positions are given and the time asked. This
# direction needs no iteration. `time_since_periapsis` gives the time from
# the last periapsis passage to a true anomaly; the time of flight is the
# difference of two such times.

# %%
t1 = ad.time_since_periapsis(f1, a, e, mu)
t2 = ad.time_since_periapsis(f2, a, e, mu)
T = ad.period(a, mu)
time_of_flight = t2 - t1
if time_of_flight < 0.0:                     # the second point is reached in the next revolution
    time_of_flight = time_of_flight + T

print(f"period          = {T:.3f} s = {T / 3600.0:.3f} h")
print(f"time to point 1 = {t1:.3f} s")
print(f"time to point 2 = {t2:.3f} s")
print(f"time of flight  = {time_of_flight:.3f} s = {time_of_flight / 3600.0:.3f} h")

# %% [markdown]
# ## 3. The same solution, step by step
#
# For each point: true anomaly to eccentric anomaly (geometry), eccentric to
# mean anomaly (Kepler's equation, used directly), mean anomaly to time
# (division by the mean motion).

# %%
n = ad.mean_motion(a, mu)                    # mean motion, rad/s

E1 = ad.f2E(f1, e)                           # eccentric anomaly of point 1
M1 = E1 - e * np.sin(E1)                     # Kepler's equation
t1_steps = M1 / n

E2 = ad.f2E(f2, e)
M2 = E2 - e * np.sin(E2)
t2_steps = M2 / n

print(f"point 1: E = {np.degrees(E1):.5f} deg, M = {np.degrees(M1):.5f} deg, t = {t1_steps:.3f} s")
print(f"point 2: E = {np.degrees(E2):.5f} deg, M = {np.degrees(M2):.5f} deg, t = {t2_steps:.3f} s")

assert np.isclose(t1_steps % T, t1) and np.isclose(t2_steps % T, t2)
print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. Independent check: propagate forward
#
# Start at the first point and let the time of flight pass, solving Kepler's
# equation in the other direction (time to position): the spacecraft must
# arrive at the second point. For the original data the second point is the
# apogee, which is reached exactly half a period after the perigee.

# %%
f_arrival = ad.propagate_kepler(a, e, f1, time_of_flight, mu)
print(f"true anomaly after the time of flight = {np.degrees(f_arrival) % 360.0:.6f} deg")

assert np.isclose(np.cos(f_arrival), np.cos(f2)) and np.isclose(np.sin(f_arrival), np.sin(f2), atol=1e-9)
if f2_deg == 180.0:
    assert np.isclose(t2, 0.5 * T)
    print("The time to apogee is half the period.")
print("Propagating from point 1 arrives at point 2.")

# %% [markdown]
# ## 5. Figure
#
# The orbit and the two points. The arc is a quarter of a turn in angle, but
# it takes much more than a quarter of the period, because the spacecraft is
# slow far from the Earth.

# %%
p = ad.semilatus_rectum(a, e)
fig, ax = ad.plot_orbit_plane(a, e, body_radius=ad.R_EARTH, label="orbit")
for f, name in ((f1, "point 1"), (f2, "point 2")):
    r = ad.orbit_radius(p, e, f)
    ax.plot(r * np.cos(f), r * np.sin(f), "o", label=name)
ax.legend(loc="upper right")
plt.show()

print(f"fraction of the turn  : {((f2 - f1) % (2.0 * np.pi)) / (2.0 * np.pi):.3f}")
print(f"fraction of the period: {time_of_flight / T:.3f}")

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "f1": f1_deg, "f2": f2_deg, "mu": mu}
results = {"t1": t1, "t2": t2, "time_of_flight": time_of_flight, "period": T}
validation.compare("L4-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Go from 0 to 90 degrees instead. Compare the time with the one from 90
#    to 180 degrees. Which law of Kepler explains the difference?
# 2. Set `f1_deg = 270.0` and `f2_deg = 90.0`: the arc that passes through
#    the perigee. Check the result with section 4.
# 3. Set `e = 0.0`. What fraction of the period does a quarter turn take?
