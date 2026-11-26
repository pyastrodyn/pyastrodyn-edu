# %% [markdown]
# # Example problem 6.4: A burn with radial and normal components
#
# **Lecture 6, the impulsive approximation.** Inspired by Schaub and
# Junkins, *Analytical Mechanics of Space Systems*; the statement and the
# numbers were written for this library.
#
# **Problem.** The spacecraft of Example problem 5.1 applies a velocity
# change with Hill components 0.05 km/s radial, 0 transverse and 0.2 km/s
# normal. Find the orbital elements after the burn.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The angles are given in degrees and converted to radians.
# The Hill basis H moves with the spacecraft: its first axis is radial (away
# from the Earth), the third is normal (along the angular momentum), and the
# second, transverse, completes the right-handed basis (in the plane of the
# orbit, towards the motion).

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
a = 9500.0                            # semi-major axis, km (original: 9500.0)
e = 0.15                              # eccentricity (original: 0.15)
i_deg = 52.0                          # inclination, degrees (original: 52.0)
raan_deg = 40.0                       # right ascension of the ascending node, degrees (original: 40.0)
argp_deg = 110.0                      # argument of periapsis, degrees (original: 110.0)
f_deg = 75.0                          # true anomaly, degrees (original: 75.0)
dv_H = np.array([0.05, 0.0, 0.2])     # velocity change in Hill components, km/s (original: [0.05, 0.0, 0.2])
mu = ad.MU_EARTH                      # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

i, raan, argp, f = np.radians([i_deg, raan_deg, argp_deg, f_deg])

# %% [markdown]
# ## 2. The state before the burn

# %%
r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, mu)
print("r_N =", r_N, "km")
print("v_N =", v_N, "km/s")

# %% [markdown]
# ## 3. Solution with the library
#
# `apply_impulse` adds the velocity change, given in Hill components, to the
# velocity, given in inertial components. The position does not change. The
# new state is then converted to elements.

# %%
v_new_N = ad.apply_impulse(r_N, v_N, dv_H)
a_new, e_new, i_new, raan_new, argp_new, f_new = ad.rv2coe(r_N, v_new_N, mu)

names = ("a (km)", "e", "i (deg)", "raan (deg)", "argp (deg)", "f (deg)")
before = (a, e, i_deg, raan_deg, argp_deg, f_deg)
after = (a_new, e_new, np.degrees(i_new), np.degrees(raan_new), np.degrees(argp_new), np.degrees(f_new))
print(f"{'element':12s} {'before':>14s} {'after':>14s} {'change':>12s}")
for name, old, new in zip(names, before, after):
    print(f"{name:12s} {old:14.6f} {new:14.6f} {new - old:+12.6f}")

# %% [markdown]
# ## 4. The same solution, step by step
#
# A vector can be added to another only if both are written in the same
# basis. The velocity is in N and the velocity change in H, so the velocity
# change must first be written in N.
#
# `dcm_HN` builds the matrix `C_HN` from the state: it takes components in N
# and returns components in H ("to H, from N"). The opposite direction is
# its inverse, `C_NH`, which for such a matrix is its transpose.

# %%
C_HN = ad.dcm_HN(r_N, v_N)                   # to H, from N
C_NH = ad.invert_dcm(C_HN)                   # to N, from H

dv_N = ad.change_basis(C_NH, dv_H)           # the same vector, components in N
v_steps_N = v_N + dv_N                       # now the sum makes sense

print("C_HN =")
print(np.round(C_HN, 6))
print("velocity change in N:", dv_N, "km/s")

assert np.allclose(v_steps_N, v_new_N)
print("Sections 3 and 4 agree.")

# %% [markdown]
# ## 5. Independent check: the speed after the burn
#
# In the Hill basis the velocity before the burn has a radial and a
# transverse component and no normal one. Adding the three components of the
# burn gives the new speed directly, by Pythagoras. It must equal the
# magnitude of the new velocity found above.

# %%
v_H = ad.change_basis(C_HN, v_N)             # velocity before the burn, components in H
speed_expected = np.sqrt((v_H[0] + dv_H[0])**2 + (v_H[1] + dv_H[1])**2 + (v_H[2] + dv_H[2])**2)

print("velocity before the burn in H:", np.round(v_H, 6), "km/s")
print(f"speed after the burn: {np.linalg.norm(v_new_N):.9f} km/s (expected {speed_expected:.9f})")

assert abs(v_H[2]) < 1e-12                   # no normal component before the burn
assert np.isclose(np.linalg.norm(v_new_N), speed_expected)
print("The speed after the burn is as expected.")

# %% [markdown]
# ## 6. Reading the result
#
# The normal component is the only one that turns the plane of the orbit:
# it changes the inclination and the node. The radial component keeps the
# plane and changes the shape of the orbit and the direction of its
# periapsis. The transverse component, zero here, is the efficient one to
# change the size of the orbit.

# %%
h_before = np.linalg.norm(ad.angular_momentum(r_N, v_N))
h_after = np.linalg.norm(ad.angular_momentum(r_N, v_new_N))
angle_between_planes = np.degrees(np.arccos(np.dot(ad.angular_momentum(r_N, v_N), ad.angular_momentum(r_N, v_new_N))
                                    / (h_before * h_after)))
print(f"angle between the planes before and after = {angle_between_planes:.4f} deg")
print(f"angular momentum: {h_before:.3f} -> {h_after:.3f} km^2/s")

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "i": i_deg, "raan": raan_deg, "argp": argp_deg, "f": f_deg, "dv_H": dv_H, "mu": mu}
results = dict(zip(("a", "e", "i", "raan", "argp", "f"), after))
validation.compare("L6-4", data, results)

# %% [markdown]
# ## Now try
#
# 1. Apply only the normal component, `dv_H = np.array([0.0, 0.0, 0.2])`.
#    Which elements change? Compare the plane change with
#    0.2 / (transverse speed), in radians.
# 2. Apply only a transverse component of 0.2 km/s. Which elements change
#    now?
# 3. The effect of a normal burn on the inclination and on the node depends
#    on where along the orbit it is made. Try `f_deg = 250.0` (so that
#    argp + f is a whole turn: the spacecraft is at the ascending node).
