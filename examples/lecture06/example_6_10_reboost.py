# %% [markdown]
# # Example problem 6.10: Reboost of a space station
#
# **Lecture 6, tangential impulses; short burn or small burn.** Inspired by
# Schaub and Junkins, *Analytical Mechanics of Space Systems*; the statement
# and the numbers were written for this library.
#
# **Problem.** A space station is on a circular orbit at 400 km altitude.
# (a) For one prograde impulse of 1 m/s find the change of the semi-major
# axis, at first order and exactly, the new eccentricity and the rise of the
# apogee. (b) Find the two impulses that raise the whole circular orbit by
# 2 km. (c) Compare the first-order formula with the exact change of the
# semi-major axis for prograde impulses of 1 m/s, 10 m/s, 100 m/s, 1 km/s
# and 2.397 km/s.
#
# This is the example of the slides "Example: reboost of a space station"
# (parts a and b) and "Effect of a small impulse on the energy" (part c) of
# Lecture 6.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The impulses are therefore in km/s: 1 m/s is 0.001 km/s.
# They are printed in m/s.

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
altitude = 400.0                               # altitude of the circular orbit, km (original: 400.0)
dv = 0.001                                     # one prograde impulse, km/s (original: 0.001)
orbit_raise = 2.0                              # raise of the whole orbit, km (original: 2.0)
dv_table = [0.001, 0.01, 0.1, 1.0, 2.397]      # impulses for part (c), km/s (original: [0.001, 0.01, 0.1, 1.0, 2.397])
R = ad.R_EARTH                                 # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH                               # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The orbit of the station

# %%
r = R + altitude
n = ad.mean_motion(r, mu)                      # rad/s
v = ad.circular_speed(r, mu)
period = ad.period(r, mu)

print(f"r = {r:.3f} km, speed = {v:.5f} km/s")
print(f"mean motion = {n:.5e} rad/s, period = {period / 60.0:.2f} min")

# %% [markdown]
# ## 3. Part (a): one prograde impulse
#
# **First order.** Differentiating the vis-viva equation at fixed position,
# a small tangential impulse changes the semi-major axis by
# da = (2 a^2 v / mu) dv. On a circular orbit v = n a, and this becomes
# da = 2 dv / n.
#
# **Exactly.** The new speed at the unchanged distance gives the new
# semi-major axis through the vis-viva equation, 1/a = 2/r - v^2/mu.
#
# The maneuver point becomes the perigee of the new orbit; half a revolution
# later the station is at the apogee, higher by twice the change of a.

# %%
a_change_first_order = 2.0 * dv / n
a_new = 1.0 / (2.0 / r - (v + dv) ** 2 / mu)
a_change_exact = a_new - r
e_new = (a_new - r) / a_new                    # the perigee is at r: r = a (1 - e)
apogee_rise = 2.0 * a_change_exact

print(f"change of a, first order = {1000.0 * a_change_first_order:.1f} m")
print(f"change of a, exact       = {1000.0 * a_change_exact:.1f} m")
print(f"new eccentricity = {e_new:.3e}")
print(f"half a revolution later the station is {apogee_rise:.3f} km higher")

# %% [markdown]
# ## 4. Part (b): raising the whole circular orbit
#
# One impulse raises only the opposite side of the orbit. To end on a
# circular orbit 2 km higher, two prograde impulses are needed, half a
# revolution apart: a Hohmann transfer between two very close circles. Each
# impulse is about n times the raise divided by 4.

# %%
raise_dv1, raise_dv2, transfer_time = ad.hohmann(r, r + orbit_raise, mu)
raise_dv_total = raise_dv1 + raise_dv2

print(f"first impulse  = {1000.0 * raise_dv1:.4f} m/s")
print(f"second impulse = {1000.0 * raise_dv2:.4f} m/s, {transfer_time / 60.0:.1f} min later")
print(f"total = {1000.0 * raise_dv_total:.4f} m/s   (n * raise / 2 = {1000.0 * n * orbit_raise / 2.0:.4f} m/s)")
print(f"rule of thumb: {1000.0 * raise_dv_total / orbit_raise:.3f} m/s per km of altitude")

# %% [markdown]
# ## 5. Part (c): how far does the first-order formula hold?
#
# The impulsive model itself is exact for an impulse of any size. The
# first-order formula da = 2 dv / n is not: it needs the impulse to be small
# compared with the orbital speed.

# %%
dv_array = np.asarray(dv_table, dtype=float)
table_exact = 1.0 / (2.0 / r - (v + dv_array) ** 2 / mu) - r
table_first_order = 2.0 * dv_array / n
table_error_percent = 100.0 * (table_first_order - table_exact) / table_exact

print(f"{'dv, km/s':>10s} {'dv / v, %':>10s} {'exact, km':>12s} {'first order, km':>16s} {'error, %':>10s}")
for one_dv, exact, first, error in zip(dv_array, table_exact, table_first_order, table_error_percent):
    print(f"{one_dv:10.3f} {100.0 * one_dv / v:10.3f} {exact:12.3f} {first:16.3f} {error:10.2f}")

# %% [markdown]
# The formula is excellent for a reboost and useless for an orbit transfer:
# there the exact relations must be used.

# %% [markdown]
# ## 6. Independent check: apply the impulses to the state
#
# Without the formulas: put the station on its orbit, add the impulses to
# the velocity, and compute the orbits from the states.

# %%
r_N, v_N = ad.coe2rv(r, 0.0, 0.0, 0.0, 0.0, 0.0, mu)

# Part (a): one impulse.
a_check, e_check, *_ = ad.rv2coe(r_N, ad.apply_impulse(r_N, v_N, np.array([0.0, dv, 0.0])), mu)
assert np.isclose(a_check - r, a_change_exact, rtol=1e-7) and np.isclose(e_check, e_new, rtol=1e-6)

# Part (b): two impulses, half a revolution apart.
v_after1_N = ad.apply_impulse(r_N, v_N, np.array([0.0, raise_dv1, 0.0]))
r_half_N, v_half_N = ad.propagate_kepler_rv(r_N, v_after1_N, transfer_time, mu)
v_after2_N = ad.apply_impulse(r_half_N, v_half_N, np.array([0.0, raise_dv2, 0.0]))
a_final, e_final, *_ = ad.rv2coe(r_half_N, v_after2_N, mu)

print(f"after one impulse: change of a {1000.0 * (a_check - r):.1f} m, e = {e_check:.3e}")
print(f"after the two impulses: altitude {a_final - R:.4f} km, e = {e_final:.1e}")
assert np.isclose(a_final, r + orbit_raise) and e_final < 1e-9
print("The states confirm the formulas.")

# %% [markdown]
# ## 7. Figure
#
# The altitude of the station during the raise of part (b): on the old
# circle, then half a revolution on the transfer orbit, then on the new
# circle.

# %%
times = np.linspace(0.0, transfer_time, 200)
r_transfer_N, _ = ad.propagate_kepler_rv(r_N, v_after1_N, times, mu)
altitude_transfer = np.linalg.norm(r_transfer_N, axis=1) - R

fig, ax = plt.subplots()
ax.plot([-20.0, 0.0], [altitude, altitude], color="C0")
ax.plot(times / 60.0, altitude_transfer, color="C0")
ax.plot([transfer_time / 60.0, transfer_time / 60.0 + 20.0], [altitude + orbit_raise] * 2, color="C0")
ax.plot([0.0, transfer_time / 60.0], [altitude, altitude + orbit_raise], "ko", label="impulses")
ax.set_xlabel("time from the first impulse, min")
ax.set_ylabel("altitude, km")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 8. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"altitude": altitude, "dv": dv, "raise": orbit_raise, "dv_table": dv_table, "R": R, "mu": mu}
results = {"mean_motion": n, "period_minutes": period / 60.0, "a_change_first_order": a_change_first_order,
           "a_change_exact": a_change_exact, "e_new": e_new, "apogee_rise": apogee_rise,
           "raise_dv1": raise_dv1, "raise_dv2": raise_dv2, "raise_dv_total": raise_dv_total,
           "table_a_change_exact": table_exact, "table_a_change_first_order": table_first_order,
           "table_error_percent": table_error_percent}
validation.compare("L6-10", data, results)

# %% [markdown]
# ## Now try
#
# 1. The station has lost 5 km of altitude. Which two impulses bring it
#    back? Compare with the rule of thumb.
# 2. In part (c) the error is about 2.5 times dv / v for the small impulses.
#    Check it on the table.
# 3. Play the game *Reboost*: for one of its missions compute the impulses
#    with this example, and compare with your result in the game.
