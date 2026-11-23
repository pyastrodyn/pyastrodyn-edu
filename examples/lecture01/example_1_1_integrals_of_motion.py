# %% [markdown]
# # Example problem 1.1: Integrals of motion from one state
#
# **Lecture 1, the integrals of motion.** Inspired by the worked examples of
# Curtis, *Orbital Mechanics for Engineering Students*; the statement and the
# numbers were written for this library.
#
# **Problem.** A spacecraft in orbit about the Earth is observed at one
# instant with the position and velocity given below. Find the magnitude of
# its specific angular momentum, its eccentricity, its specific energy and
# its semi-major axis.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Vectors are given by their components in the inertial
# basis N; their names end in `_N`.

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
r_N = np.array([5200.0, 4100.0, -2900.0])   # position, km (original: [5200.0, 4100.0, -2900.0])
v_N = np.array([-4.1, 5.9, 2.0])            # velocity, km/s (original: [-4.1, 5.9, 2.0])
mu = ad.MU_EARTH                            # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# The two-body problem has three integrals of motion: quantities that keep
# the same value along the whole orbit. One function for each.

# %%
h_N = ad.angular_momentum(r_N, v_N)          # specific angular momentum, a vector
e_N = ad.eccentricity_vector(r_N, v_N, mu)   # eccentricity vector, points to the periapsis
energy = ad.specific_energy(r_N, v_N, mu)    # specific energy, a number

h = np.linalg.norm(h_N)                      # magnitudes
e = np.linalg.norm(e_N)
a = -mu / (2.0 * energy)                     # the energy fixes the semi-major axis

print(f"h      = {h:.4f} km^2/s")
print(f"e      = {e:.8f}")
print(f"energy = {energy:.6f} km^2/s^2")
print(f"a      = {a:.4f} km")

# %% [markdown]
# ## 3. The same solution, step by step
#
# What the three functions do inside:
#
# - angular momentum: h = r x v;
# - eccentricity vector: e = (v x h) / mu - r / |r|;
# - specific energy: v^2 / 2 - mu / r.

# %%
r = np.linalg.norm(r_N)                      # distance from the centre of the Earth
v = np.linalg.norm(v_N)                      # speed

h_steps_N = np.cross(r_N, v_N)
e_steps_N = np.cross(v_N, h_steps_N) / mu - r_N / r
energy_steps = 0.5 * v**2 - mu / r

print(f"r = {r:.3f} km, v = {v:.5f} km/s")
print("h_N =", h_steps_N)
print("e_N =", e_steps_N)

assert np.allclose(h_steps_N, h_N) and np.allclose(e_steps_N, e_N) and np.isclose(energy_steps, energy)
print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. What the results say about the orbit
#
# The energy is negative and e < 1: the orbit is an ellipse. Its lowest and
# highest points follow from a and e.

# %%
print("type of orbit:", ad.orbit_type(e))
print(f"perigee altitude = {ad.periapsis_radius(a, e) - ad.R_EARTH:.1f} km")
if e < 1.0:                                  # only an ellipse has an apogee and a period
    print(f"apogee altitude  = {ad.apoapsis_radius(a, e) - ad.R_EARTH:.1f} km")
    print(f"period           = {ad.period(a, mu) / 60.0:.2f} min")
else:
    print("The orbit is open: it has no apogee and no period.")

# %% [markdown]
# ## 5. Independent check: the integrals do not change along the orbit
#
# Move the spacecraft along its orbit by integrating the equation of motion
# numerically for 3000 s, and compute the same quantities from the new
# state. They must be the same.

# %%
r_later_N, v_later_N = ad.propagate_numerical(r_N, v_N, [3000.0], mu)
r_later_N, v_later_N = r_later_N[0], v_later_N[0]      # the state at the one requested time

h_later = np.linalg.norm(ad.angular_momentum(r_later_N, v_later_N))
e_later = np.linalg.norm(ad.eccentricity_vector(r_later_N, v_later_N, mu))
energy_later = ad.specific_energy(r_later_N, v_later_N, mu)

print("position 3000 s later:", np.round(r_later_N, 3), "km")
print(f"h: {h:.6f} -> {h_later:.6f}")
print(f"e: {e:.9f} -> {e_later:.9f}")
print(f"energy: {energy:.8f} -> {energy_later:.8f}")

assert np.isclose(h_later, h) and np.isclose(e_later, e, atol=1e-9) and np.isclose(energy_later, energy)
print("The integrals of motion are conserved.")

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The results of every example problem are stored in the library, and the
# automatic tests compare them with the solution at every change of the code.
# The comparison applies only to the original data of section 1.

# %%
data = {"r_N": r_N, "v_N": v_N, "mu": mu}
results = {"h": h, "e": e, "energy": energy, "a": a}
validation.compare("L1-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Multiply the velocity by 1.1 (`v_N = 1.1 * np.array([-4.1, 5.9, 2.0])`).
#    What happens to the energy, to a and to e?
# 2. Find by trial the factor on the velocity for which the orbit stops being
#    an ellipse. Compare it with `ad.escape_speed(r, mu) / v`.
# 3. Press `F12` on `eccentricity_vector` and compare its code with section 3.
