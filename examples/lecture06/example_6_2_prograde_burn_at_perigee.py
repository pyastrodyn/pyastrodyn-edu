# %% [markdown]
# # Example problem 6.2: A prograde burn at perigee
#
# **Lecture 6, the impulsive approximation; raising and lowering the apses.**
# Inspired by the worked examples of Curtis, *Orbital Mechanics for
# Engineering Students*; the statement and the numbers were written for this
# library.
#
# **Problem.** A spacecraft is at the perigee of an orbit with perigee
# altitude 300 km and apogee altitude 800 km. It fires its engine along the
# velocity for a velocity change of 0.1 km/s. Find the new semi-major axis,
# eccentricity, and perigee and apogee altitudes.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The burn is given by its components in the Hill basis H
# of the orbit: radial (away from the Earth), transverse (in the plane of
# the orbit, towards the motion) and normal (along the angular momentum).

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
perigee_altitude = 300.0             # km (original: 300.0)
apogee_altitude = 800.0              # km (original: 800.0)
dv_H = np.array([0.0, 0.1, 0.0])     # velocity change in Hill components, km/s (original: [0.0, 0.1, 0.0])
R = ad.R_EARTH                       # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH                     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The orbit before the burn, and the state at perigee
#
# The orientation of the orbit does not matter here, so the three
# orientation angles are set to zero. The true anomaly at perigee is zero.

# %%
r_p, r_a = R + perigee_altitude, R + apogee_altitude
a = 0.5 * (r_p + r_a)
e = (r_a - r_p) / (r_a + r_p)

r_N, v_N = ad.coe2rv(a, e, 0.0, 0.0, 0.0, 0.0, mu)          # the state at perigee

print(f"before: a = {a:.3f} km, e = {e:.6f}")
print("position at perigee:", r_N, "km")
print("velocity at perigee:", v_N, "km/s")

# %% [markdown]
# ## 3. Solution with the library
#
# In the impulsive approximation the burn is so short that the position does
# not change and the velocity jumps by the velocity change. `apply_impulse`
# adds a velocity change given in Hill components. Then the new state is
# converted to elements.

# %%
v_new_N = ad.apply_impulse(r_N, v_N, dv_H)
a_new, e_new, *_ = ad.rv2coe(r_N, v_new_N, mu)

perigee_altitude_new = ad.periapsis_radius(a_new, e_new) - R
apogee_altitude_new = ad.apoapsis_radius(a_new, e_new) - R

print(f"after: a = {a_new:.3f} km, e = {e_new:.6f}")
print(f"perigee altitude = {perigee_altitude_new:.3f} km")
print(f"apogee altitude  = {apogee_altitude_new:.3f} km")

# %% [markdown]
# ## 4. The same solution, step by step
#
# At an apse the velocity is purely transverse, so a burn along the velocity
# simply adds to the speed. The new speed at the unchanged distance gives the
# new semi-major axis through the vis-viva equation, 1/a = 2/r - v^2/mu. The
# burn point stays an apse of the new orbit, so the opposite apse is at
# 2a - r.
#
# This section applies only to a purely transverse burn.

# %%
if dv_H[0] == 0.0 and dv_H[2] == 0.0:
    v_before = ad.visviva_speed(r_p, a, mu)                 # speed at perigee before the burn
    v_after = v_before + dv_H[1]                            # the burn adds to the speed
    a_steps = 1.0 / (2.0 / r_p - v_after**2 / mu)           # vis-viva, solved for a
    r_opposite = 2.0 * a_steps - r_p                        # the apse opposite to the burn point

    print(f"speed at perigee: {v_before:.6f} -> {v_after:.6f} km/s")
    print(f"new semi-major axis = {a_steps:.3f} km, opposite apse at {r_opposite - R:.3f} km altitude")
    assert np.isclose(a_steps, a_new)
    print("Sections 3 and 4 agree.")
else:
    print("The burn is not purely transverse: this section does not apply.")

# %% [markdown]
# ## 5. Independent check
#
# For a prograde burn at perigee: the perigee must not move, and the formula
# for the tangential burn that moves the opposite apse from the old to the
# new apogee radius must return the velocity change.

# %%
if dv_H[0] == 0.0 and dv_H[2] == 0.0 and dv_H[1] > 0.0:
    dv_formula = ad.apse_change_dv(r_p, r_a, R + apogee_altitude_new, mu)
    print(f"perigee altitude after the burn = {perigee_altitude_new:.6f} km")
    print(f"burn needed to raise the apogee to the new altitude = {dv_formula:.9f} km/s")
    assert np.isclose(perigee_altitude_new, perigee_altitude) and np.isclose(dv_formula, dv_H[1])
    print("The perigee did not move, and the formula returns the burn.")
else:
    print("Not a prograde burn: this check does not apply.")

# %% [markdown]
# ## 6. Figure
#
# The orbits before and after the burn, in their plane. A dot marks the
# burn point.

# %%
argp_new = ad.rv2coe(r_N, v_new_N, mu)[4]                   # direction of the new periapsis
fig, ax = ad.plot_orbit_plane(a, e, body_radius=R, label="before")
ad.plot_orbit_plane(a_new, e_new, argp_new, ax=ax, linestyle="--", label="after")
ax.plot(r_N[0], r_N[1], "ko", label="burn")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"perigee_altitude": perigee_altitude, "apogee_altitude": apogee_altitude, "dv_H": dv_H,
        "R": R, "mu": mu}
results = {"a_new": a_new, "e_new": e_new, "perigee_altitude_new": perigee_altitude_new,
           "apogee_altitude_new": apogee_altitude_new}
validation.compare("L6-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Make the burn retrograde, `dv_H = np.array([0.0, -0.1, 0.0])`. Which
#    apse moves, and where is the perigee of the new orbit?
# 2. Find by trial the prograde burn that makes the new orbit pass at the
#    geostationary altitude, 35786 km.
# 3. Give the burn a radial component, `dv_H = np.array([0.1, 0.0, 0.0])`.
#    Does the semi-major axis change? Does the perigee stay where it was?
