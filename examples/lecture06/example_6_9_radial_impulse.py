# %% [markdown]
# # Example problem 6.9: A radial impulse
#
# **Lecture 6, effect of an impulse on the energy and on the angular
# momentum.** Inspired by the worked examples of Curtis, *Orbital Mechanics
# for Engineering Students*; the statement and the numbers were written for
# this library.
#
# **Problem.** A spacecraft is on a circular orbit at 400 km altitude. An
# impulse of 0.1 km/s is applied radially outward. Find the new orbit:
# eccentricity, true anomaly of the maneuver point, semi-latus rectum,
# perigee and apogee altitudes, and the change of the semi-major axis.
#
# This is the example of the slide "Example: a radial impulse" of Lecture 6.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Angles are in radians inside the library and are
# converted to degrees to print them.

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
altitude = 400.0     # altitude of the circular orbit, km (original: 400.0)
dv_radial = 0.1      # radial impulse, positive outward, km/s (original: 0.1)
R = ad.R_EARTH       # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# Take a state on the circular orbit, add the impulse (radial is the first
# component in the Hill basis), and convert the new state to elements.

# %%
r = R + altitude
v_circular = ad.circular_speed(r, mu)

r_N, v_N = ad.coe2rv(r, 0.0, 0.0, 0.0, 0.0, 0.0, mu)                  # a point of the circular orbit
v_new_N = ad.apply_impulse(r_N, v_N, np.array([dv_radial, 0.0, 0.0]))    # radial, transverse, normal
a, e, i, raan, argp, f = ad.rv2coe(r_N, v_new_N, mu)

p = ad.semilatus_rectum(a, e)
perigee_altitude = ad.periapsis_radius(a, e) - R
apogee_altitude = ad.apoapsis_radius(a, e) - R
a_change = a - r

print(f"r = {r:.3f} km, circular speed = {v_circular:.5f} km/s")
print(f"e = {e:.6f}")
print(f"true anomaly of the maneuver point = {np.degrees(f):.4f} deg")
print(f"p = {p:.3f} km")
print(f"perigee altitude = {perigee_altitude:.2f} km, apogee altitude = {apogee_altitude:.2f} km")
print(f"change of the semi-major axis = {a_change:.4f} km")

# %% [markdown]
# ## 3. The same solution, step by step
#
# **The angular momentum does not change.** The angular momentum is r times
# the transverse velocity, and a radial impulse does not touch the
# transverse velocity. So p = h^2 / mu keeps the value it had on the circle,
# which is r.
#
# **Eccentricity and true anomaly.** On any orbit the two velocity
# components are (Part 1)
#
# v_transverse = sqrt(mu / p) (1 + e cos f),   v_radial = sqrt(mu / p) e sin f.
#
# After the impulse v_transverse is still sqrt(mu / p), so e cos f = 0: the
# maneuver point is at f = 90 degrees. Then v_radial = sqrt(mu / p) e gives
# e = dv_radial / v_circular.

# %%
p_steps = r                                  # unchanged
e_steps = dv_radial / v_circular
f_steps = 0.5 * np.pi                        # 90 degrees (270 for an inward impulse)

r_perigee = p_steps / (1.0 + e_steps)
r_apogee = p_steps / (1.0 - e_steps)
a_steps = p_steps / (1.0 - e_steps**2)

print(f"e = {e_steps:.6f}, perigee radius {r_perigee:.2f} km, apogee radius {r_apogee:.2f} km, a = {a_steps:.3f} km")
if dv_radial > 0.0:
    assert np.isclose(p_steps, p) and np.isclose(e_steps, e) and np.isclose(f_steps, f)
    assert np.isclose(a_steps, a)
    print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. Independent check: the energy changes at second order only
#
# The change of specific energy caused by an impulse is v . dv + |dv|^2 / 2.
# On a circular orbit the velocity is transverse and the impulse is radial:
# they are perpendicular, the first term vanishes, and only the small second
# term is left. This is why the size of the orbit barely changes.

# %%
energy_gain = ad.energy_change(v_N, v_new_N - v_N)
energy_gain_states = ad.specific_energy(r_N, v_new_N, mu) - ad.specific_energy(r_N, v_N, mu)

print(f"energy gained = {energy_gain:.6f} km^2/s^2; dv^2 / 2 = {0.5 * dv_radial**2:.6f}")
assert np.isclose(energy_gain, 0.5 * dv_radial**2) and np.isclose(energy_gain_states, energy_gain)

# The same velocity change given along the velocity:
a_tangential, *_ = ad.rv2coe(r_N, ad.apply_impulse(r_N, v_N, np.array([0.0, abs(dv_radial), 0.0])), mu)
print(f"change of a: {a_change:.3f} km for the radial impulse, "
      f"{a_tangential - r:.1f} km for the same impulse along the velocity")
print("A radial impulse is an inefficient way to change the size of the orbit.")

# %% [markdown]
# ## 5. Figure
#
# The altitude along the new orbit, against the angle travelled from the
# maneuver point. The orbit is elliptical, but its mean altitude is almost
# the one of the circle.

# %%
angle = np.radians(np.linspace(0.0, 360.0, 361))            # angle travelled since the maneuver
altitude_new = ad.orbit_radius(p, e, f + angle) - R

fig, ax = plt.subplots()
ax.plot(np.degrees(angle), altitude_new, label="after the impulse")
ax.axhline(altitude, color="k", linestyle="--", linewidth=0.8, label="circular orbit")
ax.set_xlabel("angle travelled from the maneuver point, degrees")
ax.set_ylabel("altitude, km")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"altitude": altitude, "dv_radial": dv_radial, "R": R, "mu": mu}
results = {"v_circular": v_circular, "e": e, "f": np.degrees(f), "p": p,
           "perigee_altitude": perigee_altitude, "apogee_altitude": apogee_altitude, "a_change": a_change}
validation.compare("L6-9", data, results)

# %% [markdown]
# ## Now try
#
# 1. Make the impulse inward, `dv_radial = -0.1`. What changes: e, the true
#    anomaly of the maneuver point, the altitudes?
# 2. From the figure: after which angle is the spacecraft highest, and after
#    which is it back at the altitude of the circle?
# 3. Which impulse along the velocity gives the same apogee altitude? Find
#    it by trial with `apply_impulse`, or with `ad.apse_change_dv`.
