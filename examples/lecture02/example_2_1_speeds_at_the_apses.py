# %% [markdown]
# # Example problem 2.1: Speeds at perigee and apogee
#
# **Lecture 2, the orbit equation and the vis-viva equation.** Inspired by
# the worked examples of Curtis, *Orbital Mechanics for Engineering
# Students*; the statement and the numbers were written for this library.
#
# **Problem.** An Earth orbit has its perigee at an altitude of 400 km and
# its apogee at an altitude of 4000 km. Find the semi-major axis, the
# eccentricity, the speeds at perigee and at apogee, and the period.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s.

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
perigee_altitude = 400.0    # km (original: 400.0)
apogee_altitude = 4000.0    # km (original: 4000.0)
R = ad.R_EARTH              # equatorial radius of the Earth, km (original: ad.R_EARTH)
mu = ad.MU_EARTH            # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Size and shape of the orbit
#
# Altitudes are measured from the surface, radii from the centre. The two
# apse radii give the semi-major axis, a = (r_p + r_a) / 2, and the
# eccentricity, e = (r_a - r_p) / (r_a + r_p).

# %%
r_p = R + perigee_altitude                   # perigee radius
r_a = R + apogee_altitude                    # apogee radius

a = 0.5 * (r_p + r_a)
e = (r_a - r_p) / (r_a + r_p)

print(f"r_p = {r_p:.3f} km, r_a = {r_a:.3f} km")
print(f"a = {a:.3f} km, e = {e:.6f}")

assert np.isclose(ad.periapsis_radius(a, e), r_p) and np.isclose(ad.apoapsis_radius(a, e), r_a)

# %% [markdown]
# ## 3. Speeds and period
#
# The vis-viva equation gives the speed at any distance r on an orbit of
# semi-major axis a: v^2 = mu (2/r - 1/a). The period depends on a only.

# %%
v_perigee = ad.visviva_speed(r_p, a, mu)
v_apogee = ad.visviva_speed(r_a, a, mu)
T = ad.period(a, mu)

print(f"v at perigee = {v_perigee:.6f} km/s")
print(f"v at apogee  = {v_apogee:.6f} km/s")
print(f"period       = {T:.3f} s = {T / 60.0:.2f} min")

# %% [markdown]
# ## 4. Independent check: the two conservation laws
#
# At the apses the velocity is perpendicular to the radius, so the angular
# momentum is simply r v: it must be the same at perigee and at apogee. The
# specific energy, v^2 / 2 - mu / r, must also be the same, and equal to
# -mu / (2a).

# %%
h_perigee, h_apogee = r_p * v_perigee, r_a * v_apogee
energy_perigee = 0.5 * v_perigee**2 - mu / r_p
energy_apogee = 0.5 * v_apogee**2 - mu / r_a

print(f"angular momentum: {h_perigee:.4f} and {h_apogee:.4f} km^2/s")
print(f"energy: {energy_perigee:.6f} and {energy_apogee:.6f} km^2/s^2; -mu/(2a) = {-mu / (2.0 * a):.6f}")

assert np.isclose(h_perigee, h_apogee) and np.isclose(energy_perigee, energy_apogee)
assert np.isclose(energy_perigee, -mu / (2.0 * a))
print("Angular momentum and energy are the same at the two apses.")

# %% [markdown]
# ## 5. Figure
#
# The orbit in its plane, with the Earth to scale, and the speed along the
# orbit as a function of the true anomaly.

# %%
fig, ax = ad.plot_orbit_plane(a, e, body_radius=R, label="orbit")
ax.plot([r_p, -r_a], [0.0, 0.0], "k.", markersize=8)        # perigee (right) and apogee (left)
ax.set_title("Orbit in its plane")

f = np.radians(np.linspace(0.0, 360.0, 361))                # true anomaly, radians
r = ad.orbit_radius(ad.semilatus_rectum(a, e), e, f)        # distance at each true anomaly
fig2, ax2 = plt.subplots()
ax2.plot(np.degrees(f), ad.visviva_speed(r, a, mu))
ax2.set_xlabel("true anomaly, degrees")
ax2.set_ylabel("speed, km/s")
ax2.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"perigee_altitude": perigee_altitude, "apogee_altitude": apogee_altitude, "R": R, "mu": mu}
results = {"a": a, "e": e, "v_perigee": v_perigee, "v_apogee": v_apogee, "period": T}
validation.compare("L2-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set the apogee altitude equal to the perigee altitude. What are e and
#    the two speeds? Compare with `ad.circular_speed(r_p, mu)`.
# 2. Raise the apogee to 35786 km (the geostationary altitude). How do the
#    speed at perigee and the period change?
# 3. The ratio v_perigee / v_apogee equals r_a / r_p. Which conservation law
#    says so?
