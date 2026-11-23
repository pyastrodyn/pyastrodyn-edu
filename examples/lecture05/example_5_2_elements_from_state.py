# %% [markdown]
# # Example problem 5.2: Orbital elements from position and velocity
#
# **Lecture 5, orbital elements and their singularities.** Inspired by
# Vallado, *Fundamentals of Astrodynamics and Applications*; the statement
# and the numbers were written for this library.
#
# **Problem.** Find the classical orbital elements of a spacecraft observed
# with the position and velocity given below.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Vectors are given by their components in the inertial
# basis N. The library returns angles in radians; they are converted to
# degrees to print them.

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
r_N = np.array([-2700.0, 6100.0, 4300.0])    # position, km (original: [-2700.0, 6100.0, 4300.0])
v_N = np.array([-5.8, -3.4, 2.2])            # velocity, km/s (original: [-5.8, -3.4, 2.2])
mu = ad.MU_EARTH                             # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# One call. The elements come back in the order a, e, i, raan, argp, f.

# %%
a, e, i, raan, argp, f = ad.rv2coe(r_N, v_N, mu)

print(f"a    = {a:.4f} km")
print(f"e    = {e:.8f}")
print(f"i    = {np.degrees(i):.6f} deg")
print(f"raan = {np.degrees(raan):.6f} deg")
print(f"argp = {np.degrees(argp):.6f} deg")
print(f"f    = {np.degrees(f):.6f} deg")

# %% [markdown]
# ## 3. The same solution, step by step
#
# Three vectors carry all the information:
#
# - the angular momentum h = r x v, perpendicular to the plane of the orbit;
# - the node vector n = k x h, where k is the third axis of N: it lies in the
#   equator and points to the ascending node;
# - the eccentricity vector e, which points to the periapsis.
#
# Size and shape come from the energy and from |e|. Each angle is the angle
# between two of these directions; an inverse cosine alone gives an angle
# between 0 and 180 degrees, so each needs a rule to choose the right half.

# %%
r = np.linalg.norm(r_N)
v = np.linalg.norm(v_N)

h_N = np.cross(r_N, v_N)                                    # angular momentum
n_N = np.cross([0.0, 0.0, 1.0], h_N)                        # node vector
e_N = np.cross(v_N, h_N) / mu - r_N / r                     # eccentricity vector
h, n, ecc = np.linalg.norm(h_N), np.linalg.norm(n_N), np.linalg.norm(e_N)

energy = 0.5 * v**2 - mu / r
a_steps = -mu / (2.0 * energy)                              # size, from the energy

i_steps = np.arccos(h_N[2] / h)                             # between h and axis 3

print("h_N =", h_N)
print("n_N =", n_N)
print("e_N =", e_N)

regular = n > 1e-10 * h and ecc > 1e-10                     # there is a node, and there is a periapsis
if regular:
    raan_steps = np.arccos(n_N[0] / n)                      # between axis 1 and the node
    if n_N[1] < 0.0:                                        # node in the half with negative second component
        raan_steps = 2.0 * np.pi - raan_steps

    argp_steps = np.arccos(np.dot(n_N, e_N) / (n * ecc))    # between the node and the periapsis
    if e_N[2] < 0.0:                                        # periapsis below the equator
        argp_steps = 2.0 * np.pi - argp_steps

    f_steps = np.arccos(np.dot(e_N, r_N) / (ecc * r))       # between the periapsis and the position
    if np.dot(r_N, v_N) < 0.0:                              # spacecraft approaching the periapsis
        f_steps = 2.0 * np.pi - f_steps

    assert np.allclose([a_steps, ecc, i_steps, raan_steps, argp_steps, f_steps], [a, e, i, raan, argp, f])
    print("Sections 2 and 3 agree.")
else:
    print("Singular case (no node, or no periapsis): these steps do not apply.")
    print("The library uses the conventions stated in the description of rv2coe.")

# %% [markdown]
# The steps above fail for a circular orbit (no periapsis: e_N is zero) and
# for an equatorial orbit (no node: n_N is zero). The library handles these
# cases with stated conventions; see the description of `rv2coe`. The
# function `rv2coe_single` is this section written as a function, with those
# cases included.

# %%
assert np.allclose(ad.rv2coe_single(r_N, v_N, mu), [a, e, i, raan, argp, f])

# %% [markdown]
# ## 4. Independent check: convert back
#
# The inverse conversion, from the elements to the state, must return the
# given position and velocity.

# %%
r_back_N, v_back_N = ad.coe2rv(a, e, i, raan, argp, f, mu)

print("position from the elements:", r_back_N, "km")
print("velocity from the elements:", v_back_N, "km/s")

assert np.allclose(r_back_N, r_N) and np.allclose(v_back_N, v_N)
print("The elements return the given state.")

# %% [markdown]
# ## 5. What the elements say about the orbit

# %%
print("type of orbit:", ad.orbit_type(e))
print(f"perigee altitude = {ad.periapsis_radius(a, e) - ad.R_EARTH:.1f} km")
if e < 1.0:                                  # only an ellipse has an apogee and a period
    print(f"apogee altitude  = {ad.apoapsis_radius(a, e) - ad.R_EARTH:.1f} km")
    print(f"period           = {ad.period(a, mu) / 60.0:.2f} min")
else:
    print("The orbit is open: it has no apogee and no period.")

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r_N": r_N, "v_N": v_N, "mu": mu}
results = {"a": a, "e": e, "i": np.degrees(i), "raan": np.degrees(raan),
           "argp": np.degrees(argp), "f": np.degrees(f)}
validation.compare("L5-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Reverse the velocity (`v_N = -np.array([-5.8, -3.4, 2.2])`). Which
#    elements change, and how?
# 2. Set the third components of position and velocity to zero. The orbit is
#    equatorial: what does section 3 do, and what does the library return?
# 3. Read `rv2coe_single` in `pyastrodyn_edu/single_state.py`, then `rv2coe`
#    in `elements.py`: the same steps for many states at once.
