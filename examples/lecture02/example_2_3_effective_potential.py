# %% [markdown]
# # Example problem 2.3: Qualitative analysis of the orbits with the effective potential energy
#
# **Lecture 2, effective potential energy and qualitative analysis of the
# orbits.** Inspired by Schaub and Junkins, *Analytical Mechanics of Space
# Systems*; the statement and the numbers were written for this library.
#
# **Problem.** Four orbits about the Earth have the same angular momentum,
# that of a semi-latus rectum of 20000 km, and eccentricities 0, 0.6, 1 and
# 1.5. Plot the effective potential energy against the distance, and the four
# energy levels. From the graph alone find the periapsis (and apoapsis)
# radius of each orbit, the radial speed of the three non-circular orbits at
# a distance of 30000 km, and the speed far from the Earth on the hyperbola.
#
# This is the example of the slide "Qualitative analysis of orbits" of
# Lecture 2: its figure is the one drawn in section 3.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Energies per unit mass are in km^2/s^2.

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
p = 20000.0              # semi-latus rectum shared by the four orbits, km (original: 20000.0)
e_elliptic = 0.6         # eccentricity of the ellipse, between 0 and 1 (original: 0.6)
e_hyperbolic = 1.5       # eccentricity of the hyperbola, above 1 (original: 1.5)
r_sample = 30000.0       # distance at which the radial speed is wanted, km (original: 30000.0)
mu = ad.MU_EARTH         # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. Solution with the library
#
# The four orbits have the same angular momentum h = sqrt(mu p), so they
# share one curve of effective potential energy,
# V_eff(r) = h^2 / (2 r^2) - mu / r (`effective_potential`). Each orbit is a
# horizontal line at its energy, -mu (1 - e^2) / (2 p).
#
# The radial kinetic energy is the distance between the line and the curve,
# T_r = energy - V_eff. The motion is possible only where the line is above
# the curve. Where they cross, the radial speed is zero: an apse.

# %%
h = np.sqrt(mu * p)                                         # specific angular momentum, km^2/s
orbits = {"circular": 0.0, "elliptic": e_elliptic, "parabolic": 1.0, "hyperbolic": e_hyperbolic}
energy = {name: -mu * (1.0 - e**2) / (2.0 * p) + 0.0 for name, e in orbits.items()}
V_min = ad.effective_potential(p, h, mu)                    # the lowest point of the curve is at r = p
V_sample = ad.effective_potential(r_sample, h, mu)


def turning_points(level):
    """Distances at which the energy level meets the curve: 2 level r^2 + 2 mu r - h^2 = 0."""
    if abs(level) < 1e-12:                                  # the parabola: the equation is of first degree
        return [h**2 / (2.0 * mu)]
    discriminant = mu**2 + 2.0 * level * h**2
    root = np.sqrt(discriminant) if discriminant > 1e-12 * mu**2 else 0.0       # zero: the line touches the curve
    return sorted(x for x in ((-mu + root) / (2.0 * level), (-mu - root) / (2.0 * level)) if x > 0.0)


apses = {name: turning_points(level) for name, level in energy.items()}
radial_speed = {name: np.sqrt(2.0 * (energy[name] - V_sample)) for name in ("elliptic", "parabolic", "hyperbolic")}
v_infinity = np.sqrt(2.0 * energy["hyperbolic"])            # far away V_eff is zero: all the energy is radial

print(f"angular momentum h = {h:.1f} km^2/s; lowest effective potential energy = {V_min:.4f} km^2/s^2 at r = p")
for name in orbits:
    points = ", ".join(f"{x:.1f}" for x in apses[name])
    print(f"{name:10s}: energy = {energy[name]:+9.4f} km^2/s^2; line and curve meet at r = {points} km")
for name, speed in radial_speed.items():
    print(f"radial speed at {r_sample:.0f} km, {name:10s}: {speed:.4f} km/s")
print(f"speed far from the Earth on the hyperbola: {v_infinity:.4f} km/s")

# %% [markdown]
# ## 3. Figure
#
# The curve, the four energy levels, and the points where they meet: A1
# (periapsis of the hyperbola), A2 (periapsis of the parabola), A3 and A3'
# (periapsis and apoapsis of the ellipse), A4 (the circular orbit). At any
# distance, the radial kinetic energy of an orbit is the height of its line
# above the curve.

# %%
colors = {"hyperbolic": "tab:red", "parabolic": "tab:purple", "elliptic": "tab:green", "circular": "black"}
labels = {"hyperbolic": r"$\varepsilon>0$", "parabolic": r"$\varepsilon=0$", "elliptic": r"$\varepsilon<0$",
          "circular": r"$\varepsilon=-\mu/(2p)$"}
points = {"A1": ("hyperbolic", 0), "A2": ("parabolic", 0), "A3": ("elliptic", 0), "A3'": ("elliptic", 1), "A4": ("circular", 0)}
r_max = 6.0 * p
r = np.linspace(0.36 * p, r_max, 2000)
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(r, ad.effective_potential(r, h, mu), color="tab:blue", lw=1.6)
ax.text(3.8 * p, ad.effective_potential(3.8 * p, h, mu) + 0.6, r"$V_{eff}(r)$", color="tab:blue", fontsize=11)
for name in ("hyperbolic", "parabolic", "elliptic", "circular"):
    ax.plot([0.0, r_max], [energy[name]] * 2, color=colors[name], lw=1.6)
    ax.text(r_max, energy[name] + 0.3, labels[name], color=colors[name], ha="right", va="bottom", fontsize=12)
for label, (name, k) in points.items():
    ax.plot(apses[name][k], energy[name], "o", color="black", ms=5)
    ax.annotate(label, (apses[name][k], energy[name]), textcoords="offset points", xytext=(4, 5), fontsize=9)
ax.set_xlim(0.0, r_max)
ax.set_ylim(V_min - 1.5, energy["hyperbolic"] + 3.0)
ax.set_xlabel("distance from the centre of the Earth $r$, km")
ax.set_ylabel(r"energy per unit mass, km$^2$/s$^2$")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Independent check: the orbit equation
#
# What was read on the graph must agree with the formulas of the conic
# sections, which do not use the effective potential energy.
#
# 1. The apses: p / (1 + e) and, for the ellipse, p / (1 - e).
# 2. The radial speed: (mu / h) e sin f, with f the true anomaly at which the
#    orbit equation gives the sample distance.
# 3. The speed far away on the hyperbola: (mu / h) sqrt(e^2 - 1).

# %%
for name, e in orbits.items():
    assert np.isclose(apses[name][0], p / (1.0 + e))
    if name in radial_speed:
        f = np.arccos((p / r_sample - 1.0) / e)             # outbound point at the sample distance
        assert np.isclose(radial_speed[name], mu / h * e * np.sin(f))
assert np.isclose(apses["elliptic"][1], p / (1.0 - e_elliptic))
assert np.isclose(v_infinity, mu / h * np.sqrt(e_hyperbolic**2 - 1.0))
assert np.isclose(V_min, -mu / (2.0 * p))
print("The graph and the orbit equation agree.")

# %% [markdown]
# ## 5. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"p": p, "e_elliptic": e_elliptic, "e_hyperbolic": e_hyperbolic, "r_sample": r_sample, "mu": mu}
results = {"h": h, "V_min": V_min, "V_sample": V_sample, "v_infinity_hyperbolic": v_infinity,
           "apoapsis_elliptic": apses["elliptic"][1]}
for name in orbits:
    results["energy_" + name] = energy[name]
    results["periapsis_" + name] = apses[name][0]
for name, speed in radial_speed.items():
    results["radial_speed_" + name] = speed
validation.compare("L2-3", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `r_sample = 60000.0`. The ellipse does not reach that distance: what
#    happens to its radial kinetic energy, and to the code?
# 2. Set `e_elliptic = 0.9`. Where does A3' go? And A3?
# 3. Halve `p`. The curve changes, because the angular momentum changes: how
#    do the lowest point and the four levels move?
