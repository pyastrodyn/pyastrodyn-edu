# %% [markdown]
# # Example problem 5.3: A quarter of a revolution with the Lagrange coefficients
#
# **Lecture 5, the Lagrange coefficients.** Inspired by the worked examples
# of Curtis, *Orbital Mechanics for Engineering Students*; the statement and
# the numbers were written for this library.
#
# **Problem.** Starting from the state of Example problem 5.1, find the four
# Lagrange coefficients for an increase of the true anomaly of 90 degrees,
# and the position and velocity there.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The angles are given in degrees and converted to radians.

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
a = 9500.0           # semi-major axis, km (original: 9500.0)
e = 0.15             # eccentricity (original: 0.15)
i_deg = 52.0         # inclination, degrees (original: 52.0)
raan_deg = 40.0      # right ascension of the ascending node, degrees (original: 40.0)
argp_deg = 110.0     # argument of periapsis, degrees (original: 110.0)
f_deg = 75.0         # true anomaly at the start, degrees (original: 75.0)
df_deg = 90.0        # increase of the true anomaly, degrees (original: 90.0)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

i, raan, argp, f, df = np.radians([i_deg, raan_deg, argp_deg, f_deg, df_deg])

# %% [markdown]
# ## 2. The initial state
#
# As in Example problem 5.1.

# %%
r0_N, v0_N = ad.coe2rv(a, e, i, raan, argp, f, mu)
print("r0_N =", r0_N, "km")
print("v0_N =", v0_N, "km/s")

# %% [markdown]
# ## 3. Solution with the library
#
# The motion stays in the plane of r0 and v0. So the position and the
# velocity at any other point of the orbit are combinations of r0 and v0:
#
# r = F r0 + G v0,   v = Fdot r0 + Gdot v0.
#
# The four numbers F, G, Fdot, Gdot are the Lagrange coefficients. They
# depend on the initial state and on the change of true anomaly only: no
# orbital element and no rotation matrix is needed.

# %%
F, G, Fdot, Gdot = ad.lagrange_coefficients(r0_N, v0_N, df, mu)
r_N, v_N = ad.propagate_lagrange(r0_N, v0_N, df, mu)

print(f"F    = {F:.10f}")
print(f"G    = {G:.6f} s")
print(f"Fdot = {Fdot:.6e} 1/s")
print(f"Gdot = {Gdot:.10f}")
print("r_N =", r_N, "km")
print("v_N =", v_N, "km/s")

# %% [markdown]
# ## 4. The same solution, step by step
#
# Three numbers of the initial state are needed: the distance r0, the radial
# speed v_r0 = (r0 . v0) / r0, and the angular momentum h, with p = h^2 / mu.
#
# **Step 1: the distance at the new point.** It is the orbit equation with
# the true anomaly increased by df, written with quantities of the initial
# state only:
#
# r = p / (1 + (p / r0 - 1) cos(df) - (h v_r0 / mu) sin(df)).
#
# **Step 2: the coefficients.**
#
# - F = 1 - (r / p)(1 - cos df)
# - G = r r0 sin(df) / h
# - Fdot = (mu / h^2) v_r0 (1 - cos df) - (mu / (h r0)) sin(df)
# - Gdot = 1 - (r0 / p)(1 - cos df)

# %%
r0 = np.linalg.norm(r0_N)                                   # distance at the start
v_r0 = np.dot(r0_N, v0_N) / r0                              # radial speed at the start
h = np.linalg.norm(np.cross(r0_N, v0_N))                    # angular momentum
p = h**2 / mu                                               # semi-latus rectum

r = p / (1.0 + (p / r0 - 1.0) * np.cos(df) - (h * v_r0 / mu) * np.sin(df))

F_steps = 1.0 - (r / p) * (1.0 - np.cos(df))
G_steps = r * r0 * np.sin(df) / h
Fdot_steps = (mu / h**2) * v_r0 * (1.0 - np.cos(df)) - (mu / (h * r0)) * np.sin(df)
Gdot_steps = 1.0 - (r0 / p) * (1.0 - np.cos(df))

r_steps_N = F_steps * r0_N + G_steps * v0_N                 # the new position
v_steps_N = Fdot_steps * r0_N + Gdot_steps * v0_N           # the new velocity

print(f"distance at the start {r0:.3f} km, at the end {r:.3f} km")
assert np.allclose([F_steps, G_steps, Fdot_steps, Gdot_steps], [F, G, Fdot, Gdot])
assert np.allclose(r_steps_N, r_N) and np.allclose(v_steps_N, v_N)
print("Sections 3 and 4 agree.")

# %% [markdown]
# ## 5. Independent check: through the orbital elements
#
# The same point is reached by keeping the five other elements and adding
# 90 degrees to the true anomaly. Also, the coefficients must satisfy
# F Gdot - G Fdot = 1, which expresses the conservation of the angular
# momentum.

# %%
r_elements_N, v_elements_N = ad.coe2rv(a, e, i, raan, argp, f + df, mu)

print("position through the elements:", r_elements_N, "km")
print(f"F Gdot - G Fdot = {F * Gdot - G * Fdot:.12f}")

assert np.allclose(r_elements_N, r_N) and np.allclose(v_elements_N, v_N)
assert np.isclose(F * Gdot - G * Fdot, 1.0)
print("Lagrange coefficients and orbital elements give the same state.")

# %% [markdown]
# ## 6. Figure
#
# The orbit in space with the initial point and the point a quarter of a
# revolution later. The whole orbit is drawn with the Lagrange coefficients
# too: 361 values of the change of true anomaly in one call.

# %%
df_all = np.radians(np.linspace(0.0, 360.0, 361))
r_orbit_N, _ = ad.propagate_lagrange(r0_N, v0_N, df_all, mu)

fig, ax = ad.plot_orbit_3d(r_orbit_N, label="orbit")
ax.plot(*r0_N, "o", label="start")
ax.plot(*r_N, "s", label=f"{df_deg:.0f} degrees later")
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "i": i_deg, "raan": raan_deg, "argp": argp_deg, "f": f_deg, "df": df_deg, "mu": mu}
results = {"F": F, "G": G, "Fdot": Fdot, "Gdot": Gdot, "r_N": r_N, "v_N": v_N}
validation.compare("L5-3", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `df_deg = 360.0`. What are the four coefficients, and why?
# 2. Set `df_deg = 180.0`. What is G? What does that say about the positions
#    at the two ends of a half revolution?
# 3. The coefficients here depend on the change of true anomaly, not on the
#    time. How would you find the time that the quarter revolution takes?
#    (Example problem 4.2.)
