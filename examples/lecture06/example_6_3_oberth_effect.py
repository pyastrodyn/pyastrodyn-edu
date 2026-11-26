# %% [markdown]
# # Example problem 6.3: The Oberth effect
#
# **Lecture 6, where to burn: the Oberth effect.** Inspired by the worked
# examples of Curtis, *Orbital Mechanics for Engineering Students*; the
# statement and the numbers were written for this library.
#
# **Problem.** A spacecraft is on a geostationary transfer orbit with perigee
# altitude 400 km and apogee altitude 35786 km. Compare a prograde velocity
# change of 0.1 km/s applied at perigee and at apogee: the specific energy
# gained, and how far the opposite apse rises.
#
# This is the example of the slide "The Oberth effect" of Lecture 6.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. A specific energy in km^2/s^2 is the same number in
# MJ/kg.

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
perigee_altitude = 400.0     # km (original: 400.0)
apogee_altitude = 35786.0    # km (original: 35786.0)
dv = 0.1                     # prograde velocity change, km/s (original: 0.1)
R = ad.R_EARTH               # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH             # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Speeds at the two apses

# %%
r_perigee = R + perigee_altitude
r_apogee = R + apogee_altitude
a = 0.5 * (r_perigee + r_apogee)
e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
v_perigee = ad.visviva_speed(r_perigee, a, mu)
v_apogee = ad.visviva_speed(r_apogee, a, mu)

print(f"a = {a:.3f} km, e = {e:.6f}")
print(f"speed at perigee = {v_perigee:.6f} km/s")
print(f"speed at apogee  = {v_apogee:.6f} km/s")

# %% [markdown]
# ## 3. Solution: the energy gained
#
# The specific energy is v^2 / 2 - mu / r. An impulsive burn changes the
# velocity and not the position, so only the kinetic part changes:
#
# change of energy = v . dv + |dv|^2 / 2.
#
# The first term is proportional to the speed at the burn point: the same
# velocity change adds more energy where the spacecraft is fastest.
# `energy_change` takes the velocity and the velocity change as vectors in
# the same basis; at an apse a prograde burn is along the velocity, so both
# are put along the first axis.

# %%
gain_perigee = ad.energy_change([v_perigee, 0.0, 0.0], [dv, 0.0, 0.0])
gain_apogee = ad.energy_change([v_apogee, 0.0, 0.0], [dv, 0.0, 0.0])
ratio = gain_perigee / gain_apogee

print(f"energy gained at perigee = {gain_perigee:.6f} km^2/s^2")
print(f"energy gained at apogee  = {gain_apogee:.6f} km^2/s^2")
print(f"ratio = {ratio:.4f}")

assert np.isclose(gain_perigee, v_perigee * dv + 0.5 * dv**2)      # the formula, written out

# %% [markdown]
# ## 4. Solution: how far the opposite apse rises
#
# A tangential burn at an apse leaves that apse where it is and moves the
# opposite one. The new speed at the unchanged distance r gives the new
# semi-major axis through the vis-viva equation, 1/a = 2/r - v^2/mu, and the
# opposite apse is then at 2a - r.

# %%
a_after_perigee_burn = 1.0 / (2.0 / r_perigee - (v_perigee + dv) ** 2 / mu)
a_after_apogee_burn = 1.0 / (2.0 / r_apogee - (v_apogee + dv) ** 2 / mu)

apogee_rise = (2.0 * a_after_perigee_burn - r_perigee) - r_apogee
perigee_rise = (2.0 * a_after_apogee_burn - r_apogee) - r_perigee

print(f"burn at perigee: the apogee rises by {apogee_rise:.1f} km")
print(f"burn at apogee:  the perigee rises by {perigee_rise:.1f} km")

# %% [markdown]
# ## 5. Independent check: from the states
#
# Without the formulas: take the state at each apse, apply the burn, and
# compute the energy and the apses of the new orbit from the new state.

# %%
energy_before = -mu / (2.0 * a)
for name, f in (("perigee", 0.0), ("apogee", np.pi)):       # true anomaly of the burn point
    r_N, v_N = ad.coe2rv(a, e, 0.0, 0.0, 0.0, f, mu)
    v_new_N = ad.apply_impulse(r_N, v_N, np.array([0.0, dv, 0.0]))     # transverse = prograde at an apse
    gain = ad.specific_energy(r_N, v_new_N, mu) - energy_before
    a_new, e_new, *_ = ad.rv2coe(r_N, v_new_N, mu)
    print(f"burn at {name}: energy gained {gain:.6f} km^2/s^2; new orbit "
          f"{ad.periapsis_radius(a_new, e_new) - R:.1f} km x {ad.apoapsis_radius(a_new, e_new) - R:.1f} km of altitude")
    if name == "perigee":
        assert np.isclose(gain, gain_perigee)
        assert np.isclose(ad.apoapsis_radius(a_new, e_new) - r_apogee, apogee_rise, rtol=1e-6)
    else:
        assert np.isclose(gain, gain_apogee)
        assert np.isclose(ad.periapsis_radius(a_new, e_new) - r_perigee, perigee_rise, rtol=1e-6)
print("Formulas and states agree.")

# %% [markdown]
# ## 6. Figure
#
# The energy gained by the same prograde burn at every point of the orbit.
# Away from the apses the burn is taken along the velocity, so the gain is
# v dv + dv^2 / 2 with the local speed v.

# %%
f_all = np.radians(np.linspace(0.0, 360.0, 361))
r_all = ad.orbit_radius(ad.semilatus_rectum(a, e), e, f_all)
v_all = ad.visviva_speed(r_all, a, mu)

fig, ax = plt.subplots()
ax.plot(np.degrees(f_all), v_all * dv + 0.5 * dv**2)
ax.set_xlabel("true anomaly of the burn point, degrees")
ax.set_ylabel("specific energy gained, km^2/s^2")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"perigee_altitude": perigee_altitude, "apogee_altitude": apogee_altitude, "dv": dv, "R": R, "mu": mu}
results = {"v_perigee": v_perigee, "v_apogee": v_apogee, "energy_gain_perigee": gain_perigee,
           "energy_gain_apogee": gain_apogee, "ratio": ratio, "apogee_rise": apogee_rise,
           "perigee_rise": perigee_rise}
validation.compare("L6-3", data, results)

# %% [markdown]
# ## Now try
#
# 1. For a very small burn the ratio of the energy gains tends to
#    v_perigee / v_apogee, which equals r_apogee / r_perigee. Set
#    `dv = 0.001` and compare.
# 2. Make the orbit circular (equal altitudes). What is the ratio?
# 3. The burn at apogee gains little energy. Why is the second burn of a
#    Hohmann transfer made there all the same? (Look at what it does to the
#    perigee.)
