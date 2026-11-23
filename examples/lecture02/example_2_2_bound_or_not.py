# %% [markdown]
# # Example problem 2.2: Bound or not?
#
# **Lecture 2, the energy constant and the conic sections.** Inspired by the
# worked examples of Curtis, *Orbital Mechanics for Engineering Students*;
# the statement and the numbers were written for this library.
#
# **Problem.** At a distance of 9000 km from the centre of the Earth a
# spacecraft has a speed of 10 km/s and a flight-path angle of 20 degrees.
# Compare its speed with the escape speed; find its specific energy, its
# speed far from the Earth, its eccentricity and the type of its orbit.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The flight-path angle is given in degrees and converted to
# radians. It is the angle of the velocity above the local horizontal:
# positive when the spacecraft is moving away from the Earth.

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
r = 9000.0                     # distance from the centre of the Earth, km (original: 9000.0)
v = 10.0                       # speed, km/s (original: 10.0)
flight_path_angle_deg = 20.0   # flight-path angle, degrees (original: 20.0)
mu = ad.MU_EARTH               # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

gamma = np.radians(flight_path_angle_deg)    # the same angle in radians

# %% [markdown]
# ## 2. Speed against escape speed, and the energy
#
# The escape speed at distance r is sqrt(2 mu / r): the speed for which the
# specific energy, v^2 / 2 - mu / r, is zero. Faster than that, the energy is
# positive and the spacecraft does not come back.

# %%
v_escape = ad.escape_speed(r, mu)
energy = 0.5 * v**2 - mu / r

print(f"speed        = {v:.6f} km/s")
print(f"escape speed = {v_escape:.6f} km/s")
print(f"energy       = {energy:.6f} km^2/s^2")
print("The spacecraft", "escapes." if energy > 0.0 else "is bound to the Earth.")

# %% [markdown]
# ## 3. Speed far from the Earth
#
# Far away the term mu / r vanishes, so all the energy is kinetic:
# v_infinity^2 / 2 = energy, that is v_infinity^2 = v^2 - v_escape^2.
# For a bound orbit there is no such speed.

# %%
if energy > 0.0:
    v_infinity = np.sqrt(v**2 - v_escape**2)
    print(f"v_infinity = {v_infinity:.6f} km/s")
else:
    v_infinity = float("nan")
    print("Bound orbit: the spacecraft never gets far from the Earth.")

# %% [markdown]
# ## 4. Eccentricity and type of orbit
#
# The angular momentum is h = r v cos(gamma): only the horizontal part of
# the velocity counts. Energy and angular momentum together give the
# eccentricity: e^2 = 1 + 2 energy h^2 / mu^2.

# %%
h = r * v * np.cos(gamma)
e = np.sqrt(1.0 + 2.0 * energy * h**2 / mu**2)
orbit_type = ad.orbit_type(e)

print(f"h = {h:.3f} km^2/s")
print(f"e = {e:.8f}")
print("type of orbit:", orbit_type)

# %% [markdown]
# ## 5. Independent check: from the state vector
#
# Build a position and a velocity that have the given distance, speed and
# flight-path angle, and compute the eccentricity as the magnitude of the
# eccentricity vector. The speed far away is recomputed from the vis-viva
# equation at a very large distance.

# %%
r_N = np.array([r, 0.0, 0.0])                                     # on axis 1
v_N = v * np.array([np.sin(gamma), np.cos(gamma), 0.0])           # radial part, horizontal part

e_check = np.linalg.norm(ad.eccentricity_vector(r_N, v_N, mu))
print(f"e from the eccentricity vector = {e_check:.8f}")
assert np.isclose(e_check, e)
assert np.isclose(np.degrees(ad.flight_path_angle_rv(r_N, v_N)), flight_path_angle_deg)

if energy > 0.0:
    a = -mu / (2.0 * energy)                                      # negative for a hyperbola
    v_far = ad.visviva_speed(1.0e12, a, mu)                       # at a very large distance
    print(f"semi-major axis = {a:.3f} km; speed at a very large distance = {v_far:.6f} km/s")
    assert np.isclose(v_far, v_infinity)
print("The two routes agree.")

# %% [markdown]
# ## 6. Figure: the effective potential
#
# The radial motion is that of a particle in the effective potential
# U_eff(r) = h^2 / (2 r^2) - mu / r. The spacecraft can only be where its
# energy is above the curve. An energy line above zero never meets the
# curve again at large r: the spacecraft escapes.

# %%
distances = np.linspace(0.3 * r, 12.0 * r, 600)
fig, ax = plt.subplots()
ax.plot(distances, ad.effective_potential(distances, h, mu), label="effective potential")
ax.axhline(energy, linestyle="--", color="C1", label="energy of the spacecraft")
ax.axhline(0.0, color="k", linewidth=0.5)
ax.plot(r, energy, "ko", label="the spacecraft now")
ax.set_ylim(-40.0, 15.0)
ax.set_xlabel("distance from the centre of the Earth, km")
ax.set_ylabel("specific energy, km^2/s^2")
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r": r, "v": v, "flight_path_angle": flight_path_angle_deg, "mu": mu}
results = {"v_escape": v_escape, "energy": energy, "v_infinity": v_infinity, "e": e, "orbit_type": orbit_type}
validation.compare("L2-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `v = 8.0`. What changes in every section? Read the lowest and
#    highest distance from the figure.
# 2. With the original speed, change the flight-path angle to 0 and to 60
#    degrees. Which results change and which do not? Why?
# 3. Find the speed for which the orbit is a parabola. What is e then?
