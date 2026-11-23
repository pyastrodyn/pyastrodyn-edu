# %% [markdown]
# # Example problem 5.1: Position and velocity from the orbital elements
#
# **Lecture 5, conversion between the state and the orbital elements.**
# Inspired by the worked examples of Curtis, *Orbital Mechanics for
# Engineering Students*; the statement and the numbers were written for this
# library.
#
# **Problem.** Find the position and the velocity, in inertial components,
# of a spacecraft with the classical orbital elements given below.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The angles are given in degrees and converted to radians.
# The name of a vector ends with the letter of the basis in which its
# components are given: `r_P` in the perifocal basis P, `r_N` in the inertial
# basis N.

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
f_deg = 75.0         # true anomaly, degrees (original: 75.0)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

i, raan, argp, f = np.radians([i_deg, raan_deg, argp_deg, f_deg])     # the same angles in radians

# %% [markdown]
# ## 2. Solution with the library
#
# One call. The elements are always in the order a, e, i, raan, argp, f.

# %%
r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, mu)

print("r_N =", r_N, "km")
print("v_N =", v_N, "km/s")

# %% [markdown]
# ## 3. The same solution, step by step
#
# **Step 1: in the plane of the orbit.** The perifocal basis P has its first
# axis towards the periapsis and its third axis along the angular momentum.
# In it the orbit is a planar curve, and position and velocity depend only on
# a, e and f:
#
# - distance from the orbit equation, r = p / (1 + e cos f), with
#   p = a (1 - e^2);
# - position r_P = r [cos f, sin f, 0];
# - velocity v_P = sqrt(mu / p) [-sin f, e + cos f, 0].

# %%
p = ad.semilatus_rectum(a, e)
r = ad.orbit_radius(p, e, f)

r_P = r * np.array([np.cos(f), np.sin(f), 0.0])
v_P = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])

print(f"p = {p:.3f} km, r = {r:.3f} km")
print("r_P =", r_P, "km")
print("v_P =", v_P, "km/s")

# %% [markdown]
# **Step 2: change of basis.** The three angles raan, i and argp say how the
# perifocal basis is oriented with respect to the inertial one. They define
# the matrix `C_NP`, which takes components in P and returns components in N
# (read the name as "to N, from P"). The vector itself does not change: only
# the basis in which its components are written.

# %%
C_NP = ad.dcm_NP(i, raan, argp)

r_steps_N = ad.change_basis(C_NP, r_P)       # the same as C_NP @ r_P
v_steps_N = ad.change_basis(C_NP, v_P)

print("C_NP =")
print(np.round(C_NP, 6))

assert np.allclose(r_steps_N, r_N) and np.allclose(v_steps_N, v_N)
print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. Independent check: three properties the state must have
#
# - its distance from the centre must satisfy the orbit equation;
# - its specific energy must equal -mu / (2a);
# - its angular momentum must make the angle i with the third axis of N.

# %%
distance = np.linalg.norm(r_N)
energy = ad.specific_energy(r_N, v_N, mu)
h_N = ad.angular_momentum(r_N, v_N)
inclination = np.degrees(np.arccos(h_N[2] / np.linalg.norm(h_N)))

print(f"distance = {distance:.3f} km (orbit equation: {r:.3f} km)")
print(f"energy = {energy:.6f} km^2/s^2 (-mu/(2a) = {-mu / (2.0 * a):.6f})")
print(f"angle between the angular momentum and axis 3 = {inclination:.6f} deg")

assert np.isclose(distance, r) and np.isclose(energy, -mu / (2.0 * a)) and np.isclose(inclination, i_deg)
print("The state has the three properties.")

# %% [markdown]
# ## 5. Figure
#
# The orbit in space, obtained by giving the true anomaly all values from 0
# to 360 degrees in one call, and the position of the spacecraft.

# %%
f_all = np.radians(np.linspace(0.0, 360.0, 361))
r_orbit_N, _ = ad.coe2rv(a, e, i, raan, argp, f_all, mu)    # 361 positions at once

fig, ax = ad.plot_orbit_3d(r_orbit_N, label="orbit")
ax.plot(*r_N, "o", label="spacecraft")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "i": i_deg, "raan": raan_deg, "argp": argp_deg, "f": f_deg, "mu": mu}
results = {"r_N": r_N, "v_N": v_N}
validation.compare("L5-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set the true anomaly to 0 and then to 180 degrees. Compare the distance
#    with the periapsis and apoapsis radii, a (1 - e) and a (1 + e).
# 2. Set the inclination to 0. Which component of the position vanishes?
#    Which angles still matter, and in which combination?
# 3. Read `coe2rv_single` in `pyastrodyn_edu/single_state.py`: it is section
#    3 written as a function. Then read `coe2rv` in `elements.py`, which does
#    the same for many states at once.
