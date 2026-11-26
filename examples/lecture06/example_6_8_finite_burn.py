# %% [markdown]
# # Example problem 6.8: Finite burn or impulse?
#
# **Lecture 6, equation of motion with thrust; finite burn or impulse.**
# Inspired by Vallado, *Fundamentals of Astrodynamics and Applications*; the
# statement and the numbers were written for this library.
#
# **Problem.** A spacecraft on a circular orbit at 400 km altitude must raise
# its apoapsis to the geostationary radius. The velocity change is the one
# of the impulsive maneuver, but it is delivered by a burn of finite
# duration, along the velocity, with constant thrust acceleration, centred on
# the point where the impulse would be given. For burn times of 1, 5, 10 and
# 20 minutes find the fraction of the orbit flown during the burn, the
# apoapsis radius reached, and the equivalent loss of velocity change (from
# the orbital energy reached).
#
# This is the example of the slide "Effect of finite burn duration: LEO to
# GTO example" of Lecture 6: its table is the one printed in section 4.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s; the burn times are given in minutes.

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
altitude = 400.0                              # altitude of the circular orbit, km (original: 400.0)
burn_minutes = [1.0, 5.0, 10.0, 20.0]         # burn times, minutes (original: [1.0, 5.0, 10.0, 20.0])
sidereal_day = ad.SIDEREAL_DAY                # rotation period of the Earth, s (original: ad.SIDEREAL_DAY)
R = ad.R_EARTH                                # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH                              # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The impulsive maneuver, as reference
#
# One prograde impulse on the circular orbit that raises the apoapsis to the
# geostationary radius: the first impulse of a Hohmann transfer.

# %%
r1 = R + altitude
r_target = ad.synchronous_radius(sidereal_day, mu)
dv = ad.hohmann(r1, r_target, mu)[0]                  # the first of the two impulses
v_circular = ad.circular_speed(r1, mu)
period = ad.period(r1, mu)

print(f"r1 = {r1:.3f} km, period = {period / 60.0:.2f} min, speed = {v_circular:.5f} km/s")
print(f"target apoapsis radius = {r_target:.1f} km")
print(f"impulse needed = {dv:.5f} km/s")

# %% [markdown]
# ## 3. One finite burn, step by step
#
# With an impulse the thrust never enters the equation of motion: the
# velocity jumps, and the motion before and after is Keplerian. A burn of
# finite duration is different: the thrust is in the equation of motion,
#
# r'' = -mu r / |r|^3 + a_T v / |v|,
#
# with the thrust acceleration a_T = dv / burn time along the velocity, and
# the equation is integrated numerically over the burn. `finite_burn` does
# this (the mass is taken as constant during the burn).
#
# The burn is centred on the point where the impulse would be given: it
# starts half a burn time before that point and ends half a burn time after.

# %%
burn_time = 60.0 * burn_minutes[1]                                     # the second burn time of the list, s

r_mid_N, v_mid_N = ad.coe2rv(r1, 0.0, 0.0, 0.0, 0.0, 0.0, mu)          # the point of the impulse
r_on_N, v_on_N = ad.propagate_kepler_rv(r_mid_N, v_mid_N, -0.5 * burn_time, mu)   # ignition: earlier on the circle
r_off_N, v_off_N = ad.finite_burn(r_on_N, v_on_N, dv, burn_time, mu)             # cutoff

a, e, *_ = ad.rv2coe(r_off_N, v_off_N, mu)                             # the orbit after the burn
print(f"burn of {burn_time / 60.0:.0f} min: thrust acceleration {1000.0 * dv / burn_time:.3f} m/s^2")
print(f"orbit after the burn: periapsis {ad.periapsis_radius(a, e):.1f} km, apoapsis {ad.apoapsis_radius(a, e):.1f} km")
print(f"the impulse would give: periapsis {r1:.1f} km, apoapsis {r_target:.1f} km")

# %% [markdown]
# The apoapsis is lower than the target although the engine delivered the
# same velocity change. During a long burn the spacecraft moves along its
# orbit: the thrust is not all applied at the best point, and not all along
# the direction the impulse would have. Part of the velocity change is
# "lost". The loss is measured through the orbital energy: the equivalent
# impulse is the one that, given at r1, would produce the same energy.

# %%
energy_reached = ad.specific_energy(r_off_N, v_off_N, mu)
dv_equivalent = np.sqrt(2.0 * (energy_reached + mu / r1)) - v_circular     # vis-viva at r1
print(f"equivalent impulse = {dv_equivalent:.5f} km/s: loss {100.0 * (1.0 - dv_equivalent / dv):.3f} %")

# %% [markdown]
# ## 4. Solution: the table for all the burn times

# %%
fractions, apoapses, losses = [], [], []
for minutes in burn_minutes:
    burn_time = 60.0 * minutes
    r_on_N, v_on_N = ad.propagate_kepler_rv(r_mid_N, v_mid_N, -0.5 * burn_time, mu)
    r_off_N, v_off_N = ad.finite_burn(r_on_N, v_on_N, dv, burn_time, mu)
    a, e, *_ = ad.rv2coe(r_off_N, v_off_N, mu)
    dv_equivalent = np.sqrt(2.0 * (ad.specific_energy(r_off_N, v_off_N, mu) + mu / r1)) - v_circular
    fractions.append(100.0 * burn_time / period)
    apoapses.append(ad.apoapsis_radius(a, e))
    losses.append(100.0 * (1.0 - dv_equivalent / dv))
fractions, apoapses, losses = np.array(fractions), np.array(apoapses), np.array(losses)

print(f"{'burn time, min':>16s} {'orbit flown, %':>16s} {'apoapsis, km':>14s} {'loss of dv, %':>15s}")
print(f"{'impulse':>16s} {0.0:16.1f} {r_target:14.0f} {0.0:15.2f}")
for minutes, fraction, apoapsis, loss in zip(burn_minutes, fractions, apoapses, losses):
    print(f"{minutes:16.0f} {fraction:16.1f} {apoapsis:14.0f} {loss:15.2f}")

# %% [markdown]
# If the burn time is much shorter than the orbital period, the burn can be
# modelled as an impulse. This is the case for chemical propulsion.

# %% [markdown]
# ## 5. Independent check: a very short burn is an impulse
#
# As the burn time goes to zero at fixed velocity change, the result of the
# integration must tend to the impulsive one.

# %%
r_off_N, v_off_N = ad.finite_burn(r_mid_N, v_mid_N, dv, 1.0e-3, mu)       # a burn of one millisecond
a, e, *_ = ad.rv2coe(r_off_N, v_off_N, mu)
print(f"burn of 1 ms: apoapsis {ad.apoapsis_radius(a, e):.3f} km (impulse: {r_target:.3f} km)")
assert np.isclose(ad.apoapsis_radius(a, e), r_target, rtol=1e-6)
assert np.all(np.diff(apoapses) < 0.0) and np.all(np.diff(losses) > 0.0)   # longer burns do worse
print("The finite burn tends to the impulse, and longer burns lose more.")

# %% [markdown]
# ## 6. Figure
#
# The loss against the burn time, as a fraction of the period.

# %%
many_minutes = np.linspace(0.5, 30.0, 40)
many_losses = []
for minutes in many_minutes:
    burn_time = 60.0 * minutes
    r_on_N, v_on_N = ad.propagate_kepler_rv(r_mid_N, v_mid_N, -0.5 * burn_time, mu)
    r_off_N, v_off_N = ad.finite_burn(r_on_N, v_on_N, dv, burn_time, mu)
    dv_equivalent = np.sqrt(2.0 * (ad.specific_energy(r_off_N, v_off_N, mu) + mu / r1)) - v_circular
    many_losses.append(100.0 * (1.0 - dv_equivalent / dv))

fig, ax = plt.subplots()
ax.plot(100.0 * 60.0 * many_minutes / period, many_losses)
ax.plot(fractions, losses, "ko", label="the burn times of the problem")
ax.set_xlabel("burn time, % of the orbital period")
ax.set_ylabel("equivalent loss of velocity change, %")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"altitude": altitude, "burn_minutes": burn_minutes, "sidereal_day": sidereal_day, "R": R, "mu": mu}
results = {"dv": dv, "r_target": r_target, "period_minutes": period / 60.0,
           "orbit_fraction_percent": fractions, "apoapsis": apoapses, "dv_loss_percent": losses}
validation.compare("L6-8", data, results)

# %% [markdown]
# ## Now try
#
# 1. The loss grows roughly with the square of the burn time. Check it on
#    the table: from 5 to 10 minutes, and from 10 to 20.
# 2. Start from a higher orbit, `altitude = 20000.0`, where the period is
#    much longer. What happens to the losses for the same burn times?
# 3. An electric thruster gives about 0.1 mm/s^2. How long would it take to
#    deliver this velocity change? Is the impulsive model usable?
