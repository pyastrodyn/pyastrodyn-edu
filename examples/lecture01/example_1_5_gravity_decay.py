# %% [markdown]
# # Example problem 1.5: The decay of the gravitational acceleration with the distance
#
# **Lecture 1, gravitational acceleration decay.** Inspired by the worked
# examples of Curtis, *Orbital Mechanics for Engineering Students*; the
# statement and the numbers were written for this library.
#
# **Problem.** Find the gravitational acceleration of the Earth at its mean
# radius (6371 km), on a low orbit at an altitude of 500 km and at the
# geostationary radius (42164 km), and the last two as percentages of the
# first. Find the distance at which the acceleration is half the surface
# value. Plot the acceleration against the distance.
#
# This is the example of the slide "Gravitational acceleration decay" of
# Lecture 1: its figure is the one drawn in section 3.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s; the accelerations are printed and plotted in m/s^2.

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
#
# The slide uses the mean radius of the Earth, 6371 km. (The equatorial
# radius, `ad.R_EARTH`, is 6378.137 km.)

# %%
r_surface = 6371.0       # mean radius of the Earth, km (original: 6371.0)
altitude_low = 500.0     # altitude of the low orbit, km (original: 500.0)
r_geo = 42164.0          # geostationary radius, km (original: 42164.0)
mu = ad.MU_EARTH         # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# `gravitational_acceleration` returns mu / r^2. With km and s the result is
# in km/s^2: multiply by 1000 to have m/s^2.

# %%
r_low = r_surface + altitude_low
g_surface = ad.gravitational_acceleration(r_surface, mu) * 1e3      # m/s^2
g_low = ad.gravitational_acceleration(r_low, mu) * 1e3
g_geo = ad.gravitational_acceleration(r_geo, mu) * 1e3
ratio_low = 100.0 * g_low / g_surface
ratio_geo = 100.0 * g_geo / g_surface
r_half = np.sqrt(2.0) * r_surface                                   # half the surface value: r^2 is doubled

print(f"at the surface,   r = {r_surface:8.1f} km: g = {g_surface:.5f} m/s^2")
print(f"on the low orbit, r = {r_low:8.1f} km: g = {g_low:.5f} m/s^2 = {ratio_low:.1f} % of the surface value")
print(f"geostationary,    r = {r_geo:8.1f} km: g = {g_geo:.5f} m/s^2 = {ratio_geo:.2f} % of the surface value")
print(f"g is half the surface value at r = {r_half:.1f} km, i.e. at an altitude of {r_half - r_surface:.1f} km")

# %% [markdown]
# ## 3. Figure
#
# The acceleration from the surface to beyond the geostationary radius. The
# ticks of the horizontal axis are multiples of the radius of the Earth: at
# twice the radius the acceleration is one fourth, at three times one ninth.

# %%
r = np.linspace(r_surface, 7.0 * r_surface, 500)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(r, ad.gravitational_acceleration(r, mu) * 1e3, color="tab:blue")
for radius, value, name in ((r_surface, g_surface, "surface"), (r_low, g_low, "low orbit"), (r_geo, g_geo, "geostationary")):
    ax.plot(radius, value, "o", color="black", ms=4)
    side = "right" if radius > 5.0 * r_surface else "left"            # the last name is written to the left
    ax.annotate(name, (radius, value), textcoords="offset points", xytext=(-8 if side == "right" else 8, 6),
                ha=side, fontsize=9)
ax.set_xticks(r_surface * np.arange(1, 8))
ax.set_xlim(0.9 * r_surface, 7.0 * r_surface)
ax.set_ylim(0.0, 10.5)
ax.set_xlabel("distance from the centre of the Earth, km")
ax.set_ylabel("gravitational acceleration, m/s$^2$")
ax.grid(True, alpha=0.4)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Independent checks
#
# 1. The inverse-square law: the ratio of two accelerations is the inverse
#    ratio of the squares of the distances.
# 2. The acceleration is the magnitude of the right-hand side of the equation
#    of motion (`twobody_eom`), whatever the direction of the position.

# %%
assert np.isclose(ratio_low, 100.0 * (r_surface / r_low) ** 2)
assert np.isclose(ratio_geo, 100.0 * (r_surface / r_geo) ** 2)
direction = np.array([0.36, -0.48, 0.8])                            # a unit vector, any direction
state_N = np.concatenate((r_geo * direction, [0.0, 0.0, 0.0]))      # position and (zero) velocity
acceleration_N = ad.twobody_eom(0.0, state_N, mu)[3:]
assert np.isclose(np.linalg.norm(acceleration_N) * 1e3, g_geo)
assert np.isclose(ad.gravitational_acceleration(r_half, mu) * 1e3, 0.5 * g_surface)
print("The checks agree.")

# %% [markdown]
# ## 5. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r_surface": r_surface, "altitude_low": altitude_low, "r_geo": r_geo, "mu": mu}
results = {"g_surface": g_surface, "g_low": g_low, "g_geo": g_geo, "ratio_low": ratio_low, "ratio_geo": ratio_geo,
           "r_half": r_half}
validation.compare("L1-5", data, results)

# %% [markdown]
# ## Now try
#
# 1. Astronauts on a low orbit float. Is it because gravity is small there?
#    Look at `ratio_low`.
# 2. Set `altitude_low = 20200.0` (navigation satellites). What percentage of
#    the surface value is left?
# 3. Replace `mu` with the gravitational parameter of the Moon,
#    4902.8 km^3/s^2, and `r_surface` with 1737.4 km: what is the
#    acceleration at the surface of the Moon?
