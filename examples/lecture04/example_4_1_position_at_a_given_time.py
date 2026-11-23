# %% [markdown]
# # Example problem 4.1: Where is the spacecraft 40 minutes after perigee?
#
# **Lecture 4, position as a function of time: Kepler's equation.** Inspired
# by the worked examples of Curtis, *Orbital Mechanics for Engineering
# Students*; the statement and the numbers were written for this library.
#
# **Problem.** An Earth orbit has a semi-major axis of 12000 km and an
# eccentricity of 0.35. Find the mean, eccentric and true anomaly, and the
# distance from the centre of the Earth, 2400 s after the perigee passage.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Inside the library all angles are in radians; they are
# converted to degrees only to print them.

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
a = 12000.0          # semi-major axis, km (original: 12000.0)
e = 0.35             # eccentricity (original: 0.35)
t = 2400.0           # time since the perigee passage, s (original: 2400.0)
mu = ad.MU_EARTH     # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# Three anomalies, three steps:
#
# 1. the mean anomaly grows uniformly with time, M = n t, where
#    n = sqrt(mu / a^3) is the mean motion;
# 2. Kepler's equation, M = E - e sin E, gives the eccentric anomaly E. It
#    cannot be solved for E in closed form: `M2E` solves it numerically;
# 3. the true anomaly f follows from E by geometry.

# %%
n = ad.mean_motion(a, mu)                    # rad/s
M = n * t                                    # mean anomaly, rad
E = ad.M2E(M, e)                             # eccentric anomaly, rad
f = ad.E2f(E, e)                             # true anomaly, rad
r = ad.orbit_radius(ad.semilatus_rectum(a, e), e, f)      # distance, from the orbit equation

print(f"period = {ad.period(a, mu):.2f} s")
print(f"M = {np.degrees(M):.6f} deg")
print(f"E = {np.degrees(E):.6f} deg")
print(f"f = {np.degrees(f):.6f} deg")
print(f"r = {r:.3f} km")

# %% [markdown]
# ## 3. Kepler's equation, step by step
#
# Newton's method: start from a guess of E and correct it repeatedly with
#
# E_new = E - (E - e sin E - M) / (1 - e cos E)
#
# until the correction is negligible. A few iterations are enough.

# %%
E_k = M                                      # first guess
for k in range(1, 20):
    correction = (E_k - e * np.sin(E_k) - M) / (1.0 - e * np.cos(E_k))
    E_k = E_k - correction
    print(f"iteration {k}: E = {np.degrees(E_k):.10f} deg, correction = {correction:+.2e} rad")
    if abs(correction) < 1e-13:
        break

assert np.isclose(E_k, E)

# True anomaly from the eccentric anomaly: tan(f/2) = sqrt((1+e)/(1-e))
# tan(E/2).
f_steps = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(0.5 * E_k))
f_steps = f_steps % (2.0 * np.pi)            # in [0, 2 pi)
r_steps = a * (1.0 - e * np.cos(E_k))        # the distance, directly from E

assert np.isclose(f_steps, f % (2.0 * np.pi)) and np.isclose(r_steps, r)
print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. Independent check: numerical integration
#
# A route that does not use Kepler's equation at all. Put the spacecraft at
# perigee (true anomaly zero; the orientation of the orbit does not matter,
# so the three orientation angles are set to zero), integrate the equation
# of motion for the time t, and look at the distance and the true anomaly.

# %%
r0_N, v0_N = ad.coe2rv(a, e, 0.0, 0.0, 0.0, 0.0, mu)        # the state at perigee
r_N, v_N = ad.propagate_numerical(r0_N, v0_N, [t], mu)
r_numerical = np.linalg.norm(r_N[0])
f_numerical = ad.rv2coe(r_N[0], v_N[0], mu)[5]              # the sixth element is f

print(f"numerical integration: r = {r_numerical:.3f} km, f = {np.degrees(f_numerical):.6f} deg")
assert np.isclose(r_numerical, r) and np.isclose(f_numerical, f % (2.0 * np.pi))
print("Kepler's equation and numerical integration agree.")

# %% [markdown]
# ## 5. Figure
#
# The orbit in its plane, with the perigee on the right and the position
# after the time t.

# %%
fig, ax = ad.plot_orbit_plane(a, e, body_radius=ad.R_EARTH, label="orbit")
ax.plot(r * np.cos(f), r * np.sin(f), "o", label=f"after {t:.0f} s")
ax.plot([0.0, r * np.cos(f)], [0.0, r * np.sin(f)], "k:", linewidth=0.8)
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"a": a, "e": e, "t": t, "mu": mu}
results = {"M": np.degrees(M), "E": np.degrees(E), "f": np.degrees(f), "r": r}
validation.compare("L4-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `t` to half the period. What are M, E and f? And at a quarter of
#    the period: is f equal to 90 degrees? Why not?
# 2. Set `e = 0.9`. How many iterations does Newton's method need now?
# 3. Set `e = 0.0`. What is the relation between the three anomalies?
