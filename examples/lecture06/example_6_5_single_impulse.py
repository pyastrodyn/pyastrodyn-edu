# %% [markdown]
# # Example problem 6.5: One impulse between two intersecting orbits
#
# **Lecture 6, general single impulse: the velocity triangle.** Inspired by
# the worked examples of Curtis, *Orbital Mechanics for Engineering
# Students*; the statement and the numbers were written for this library.
#
# **Problem.** A spacecraft on a circular orbit of radius 7000 km must move,
# with one impulse, onto a coplanar orbit with semi-major axis 8400 km and
# eccentricity 0.2. Check that the orbits intersect; at the crossing point
# where the new orbit is outbound find its true anomaly, the two speeds, the
# flight-path angle of the new orbit and the velocity change.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Angles are in radians inside the library and are
# converted to degrees to print them. "Outbound" means moving away from the
# Earth: between the periapsis and the apoapsis.

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
r1 = 7000.0          # radius of the initial circular orbit, km (original: 7000.0)
a2 = 8400.0          # semi-major axis of the new orbit, km (original: 8400.0)
e2 = 0.2             # eccentricity of the new orbit (original: 0.2)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Do the orbits intersect?
#
# One impulse changes the velocity and not the position, so it can move the
# spacecraft from one orbit to another only at a point that belongs to both.
# Here the first orbit is a circle: it meets the new orbit if its radius is
# between the periapsis and the apoapsis radius of the new orbit.

# %%
intersect = bool(ad.orbits_intersect(r1, 0.0, 0.0, a2, e2, 0.0))

print(f"new orbit: periapsis {ad.periapsis_radius(a2, e2):.1f} km, apoapsis {ad.apoapsis_radius(a2, e2):.1f} km")
print(f"circle of radius {r1:.1f} km: the orbits intersect: {intersect}")

assert intersect, "The orbits do not intersect: one impulse cannot join them. Change the data."

# %% [markdown]
# ## 3. The crossing point
#
# On the new orbit the distance is r = p / (1 + e cos f). Setting r = r1
# gives the true anomaly of the crossing point: cos f = (p / r1 - 1) / e.
# The inverse cosine gives the solution between 0 and 180 degrees, which is
# the outbound one; the other crossing point is at 360 degrees minus that.

# %%
p2 = ad.semilatus_rectum(a2, e2)
f2 = np.arccos(np.clip((p2 / r1 - 1.0) / e2, -1.0, 1.0))    # clip: guards against rounding at an apse

print(f"p2 = {p2:.3f} km")
print(f"true anomaly of the crossing point on the new orbit = {np.degrees(f2):.6f} deg")
assert np.isclose(ad.orbit_radius(p2, e2, f2), r1)

# %% [markdown]
# ## 4. Solution: the velocity triangle
#
# At the crossing point the two velocities differ in magnitude and in
# direction. The speeds come from the vis-viva equation. The direction of a
# velocity in the plane of the orbit is given by its flight-path angle, the
# angle above the local horizontal: zero on the circle, gamma2 on the new
# orbit. The velocity change closes the triangle formed by the two
# velocities; the law of cosines gives its magnitude:
#
# dv^2 = v1^2 + v2^2 - 2 v1 v2 cos(gamma2 - gamma1).

# %%
v1 = ad.circular_speed(r1, mu)               # speed on the circle
v2 = ad.visviva_speed(r1, a2, mu)            # speed on the new orbit at the same distance
gamma2 = ad.flight_path_angle(e2, f2)        # flight-path angle of the new orbit there
dv = ad.single_impulse_dv(v1, v2, gamma2)    # the angle between the velocities is gamma2 - 0

print(f"v1 = {v1:.6f} km/s")
print(f"v2 = {v2:.6f} km/s")
print(f"flight-path angle of the new orbit = {np.degrees(gamma2):.6f} deg")
print(f"velocity change = {dv:.6f} km/s")

assert np.isclose(dv, np.sqrt(v1**2 + v2**2 - 2.0 * v1 * v2 * np.cos(gamma2)))    # the formula, written out

# %% [markdown]
# ## 5. Independent check: the difference of the velocity vectors
#
# Without the triangle formula: build the two states at the common point
# and subtract the velocities. Both orbits are put in the reference plane.
# The new orbit has its periapsis on the first axis; the point of the circle
# in the same direction as the crossing point is at the angle f2 from that
# axis.

# %%
r_circle_N, v_circle_N = ad.coe2rv(r1, 0.0, 0.0, 0.0, 0.0, f2, mu)     # on the circle, at angle f2
r_new_N, v_new_N = ad.coe2rv(a2, e2, 0.0, 0.0, 0.0, f2, mu)            # on the new orbit, true anomaly f2

dv_N = v_new_N - v_circle_N
print("common point:", np.round(r_new_N, 3), "km")
print("velocity change vector:", np.round(dv_N, 6), "km/s")
print(f"its magnitude = {np.linalg.norm(dv_N):.6f} km/s")

assert np.allclose(r_circle_N, r_new_N)      # it is the same point
assert np.isclose(np.linalg.norm(dv_N), dv)
print("Triangle formula and vector difference agree.")

# The burn in Hill components of the circular orbit: radial, transverse,
# normal.
dv_H = ad.change_basis(ad.dcm_HN(r_circle_N, v_circle_N), dv_N)
print("velocity change in Hill components:", np.round(dv_H, 6), "km/s")

# %% [markdown]
# ## 6. Figure
#
# The two orbits and the crossing point where the burn is made.

# %%
fig, ax = ad.plot_orbit_plane(r1, 0.0, body_radius=ad.R_EARTH, label="initial orbit")
ad.plot_orbit_plane(a2, e2, ax=ax, linestyle="--", label="new orbit")
ax.plot(r_new_N[0], r_new_N[1], "ko", label="burn")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r1": r1, "a2": a2, "e2": e2, "mu": mu}
results = {"intersect": intersect, "f2": np.degrees(f2), "v1": v1, "v2": v2,
           "flight_path_angle_2": np.degrees(gamma2), "dv": dv}
validation.compare("L6-5", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `r1 = 6720.0`, the periapsis radius of the new orbit. Where is the
#    crossing point, what is the flight-path angle, and how does the
#    velocity change compare with the original one?
# 2. Set `r1 = 6000.0`. What happens, and why?
# 3. At the other crossing point (true anomaly 360 degrees minus f2) the new
#    orbit is inbound. Is the velocity change different there?
