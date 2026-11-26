# %% [markdown]
# # Example problem 6.6: Two impulses, three Keplerian arcs
#
# **Lecture 6, the impulsive approximation: motion with one or multiple
# burns.** Inspired by the worked examples of Curtis, *Orbital Mechanics for
# Engineering Students*; the statement and the numbers were written for this
# library.
#
# **Problem.** A spacecraft moves in one plane; the polar angle theta is
# measured from a fixed axis. It starts at theta = 215 degrees on a circular
# orbit of radius 8000 km. At theta = 360 degrees a first impulse puts it on
# an orbit with eccentricity 0.4 and argument of periapsis -40 degrees. At
# theta = 150 degrees a second impulse puts it on an orbit with eccentricity
# 0.2 and argument of periapsis 100 degrees, which it follows up to
# theta = 290 degrees. Find the two orbits, the two impulses in Hill
# components (radial, transverse, normal), their magnitudes, the times of the
# two burns and of the end, and the final state.
#
# This is the example of the slide "R2BP motion with one or multiple burns"
# of Lecture 6: the figure of that slide is redrawn in section 7 from the
# computed trajectory.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. The angles are given in degrees and converted to radians.
# Everything happens in the reference plane (inclination zero), so the
# argument of periapsis is the polar angle of the periapsis, and the polar
# angle of the spacecraft is theta = argument of periapsis + true anomaly.

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
r0 = 8000.0               # radius of the initial circular orbit, km (original: 8000.0)
theta_start_deg = 215.0   # polar angle at the start, degrees (original: 215.0)
e1 = 0.4                  # eccentricity of the orbit after the first impulse (original: 0.4)
argp1_deg = -40.0         # its argument of periapsis, degrees (original: -40.0)
theta_2_deg = 150.0       # polar angle of the second impulse, degrees (original: 150.0)
e2 = 0.2                  # eccentricity of the orbit after the second impulse (original: 0.2)
argp2_deg = 100.0         # its argument of periapsis, degrees (original: 100.0)
theta_end_deg = 290.0     # polar angle at the end, degrees (original: 290.0)
mu = ad.MU_EARTH          # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

theta_start, argp1, theta_2, argp2, theta_end = np.radians(
    [theta_start_deg, argp1_deg, theta_2_deg, argp2_deg, theta_end_deg])
TWO_PI = 2.0 * np.pi

# %% [markdown]
# ## 2. The three orbits
#
# An impulse does not change the position. So the orbit after an impulse
# passes through the burn point: with its eccentricity and its argument of
# periapsis given, the orbit equation at the burn point,
# r = p / (1 + e cos f), gives its semi-latus rectum p.

# %%
# Orbit 1 passes through the first burn point: distance r0, polar angle 0.
f1_start = (0.0 - argp1) % TWO_PI                 # true anomaly on orbit 1 at the first burn
p1 = r0 * (1.0 + e1 * np.cos(f1_start))
a1 = p1 / (1.0 - e1**2)

# The second burn point is on orbit 1, at the polar angle theta_2.
f1_end = (theta_2 - argp1) % TWO_PI               # true anomaly on orbit 1 at the second burn
r_burn2 = ad.orbit_radius(p1, e1, f1_end)

# Orbit 2 passes through the second burn point.
f2_start = (theta_2 - argp2) % TWO_PI             # true anomaly on orbit 2 at the second burn
p2 = r_burn2 * (1.0 + e2 * np.cos(f2_start))
a2 = p2 / (1.0 - e2**2)
f2_end = (theta_end - argp2) % TWO_PI             # true anomaly on orbit 2 at the end

print(f"orbit 1: p = {p1:.3f} km ({p1 / r0:.3f} r0), a = {a1:.3f} km")
print(f"second burn point: distance {r_burn2:.3f} km ({r_burn2 / r0:.3f} r0)")
print(f"orbit 2: p = {p2:.3f} km ({p2 / r0:.3f} r0), a = {a2:.3f} km")

# %% [markdown]
# ## 3. The two impulses
#
# At each burn point compute the velocity on the orbit before and on the
# orbit after: the impulse is their difference. `coe2rv` gives the state from
# the elements; with inclination and node set to zero, the orbit lies in the
# reference plane. The impulse is then written in the Hill basis of the orbit
# before the burn.

# %%
# First burn: from the circle to orbit 1.
r_burn1_N, v_before1_N = ad.coe2rv(r0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
_, v_after1_N = ad.coe2rv(a1, e1, 0.0, 0.0, argp1, f1_start, mu)
dv1_N = v_after1_N - v_before1_N
dv1_H = ad.change_basis(ad.dcm_HN(r_burn1_N, v_before1_N), dv1_N)      # radial, transverse, normal

# Second burn: from orbit 1 to orbit 2.
r_burn2_N, v_before2_N = ad.coe2rv(a1, e1, 0.0, 0.0, argp1, f1_end, mu)
_, v_after2_N = ad.coe2rv(a2, e2, 0.0, 0.0, argp2, f2_start, mu)
dv2_N = v_after2_N - v_before2_N
dv2_H = ad.change_basis(ad.dcm_HN(r_burn2_N, v_before2_N), dv2_N)

dv1, dv2 = np.linalg.norm(dv1_H), np.linalg.norm(dv2_H)
dv_total = dv1 + dv2

print("first impulse, Hill components :", np.round(dv1_H, 6), f"km/s, magnitude {dv1:.6f} km/s")
print("second impulse, Hill components:", np.round(dv2_H, 6), f"km/s, magnitude {dv2:.6f} km/s")
print(f"total cost = {dv_total:.6f} km/s")

# %% [markdown]
# ## 4. The times
#
# On each arc the time follows from Kepler's equation (Lecture 4):
# `time_since_periapsis` at the two ends of the arc.

# %%
period_0 = ad.period(r0, mu)
t1 = (TWO_PI - theta_start) / TWO_PI * period_0                # on the circle the motion is uniform

coast_1 = (ad.time_since_periapsis(f1_end, a1, e1, mu)
           - ad.time_since_periapsis(f1_start, a1, e1, mu)) % ad.period(a1, mu)
t2 = t1 + coast_1

coast_2 = (ad.time_since_periapsis(f2_end, a2, e2, mu)
           - ad.time_since_periapsis(f2_start, a2, e2, mu)) % ad.period(a2, mu)
t_end = t2 + coast_2

print(f"first burn at  t1 = {t1:.3f} s")
print(f"second burn at t2 = {t2:.3f} s   (arc 1 lasts {coast_1:.3f} s)")
print(f"end at      t_end = {t_end:.3f} s   (arc 2 lasts {coast_2:.3f} s)")

# %% [markdown]
# ## 5. Analytical propagation, arc by arc
#
# The motion is a sequence of Keplerian arcs joined at the impulses. On each
# arc the solution is exact (Kepler's equation); at each burn time the
# impulse is added to the velocity, and the next arc starts from the new
# state.

# %%
r_N, v_N = ad.coe2rv(r0, 0.0, 0.0, 0.0, 0.0, theta_start, mu)     # the state at the start

r_N, v_N = ad.propagate_kepler_rv(r_N, v_N, t1, mu)               # arc 0, up to the first burn
v_N = ad.apply_impulse(r_N, v_N, dv1_H)                           # the velocity jumps
r_N, v_N = ad.propagate_kepler_rv(r_N, v_N, t2 - t1, mu)          # arc 1, up to the second burn
v_N = ad.apply_impulse(r_N, v_N, dv2_H)                           # the velocity jumps
r_end_N, v_end_N = ad.propagate_kepler_rv(r_N, v_N, t_end - t2, mu)   # arc 2

print("final position:", r_end_N, "km")
print("final velocity:", v_end_N, "km/s")

# The final state must be the point of orbit 2 at the polar angle theta_end.
r_expected_N, v_expected_N = ad.coe2rv(a2, e2, 0.0, 0.0, argp2, f2_end, mu)
assert np.allclose(r_end_N, r_expected_N, atol=1e-6) and np.allclose(v_end_N, v_expected_N, atol=1e-9)
print("The spacecraft arrives where orbit 2 says it must.")

# %% [markdown]
# ## 6. Independent check: numerical propagation, arc by arc
#
# The same logic with a numerical integrator, which is the one that still
# works when other perturbing accelerations are added (Lectures 9 and 10):
#
# 1. integrate the equation of motion and **stop exactly** at the burn time;
# 2. **add** the impulse to the velocity. The impulse is never given to the
#    integrator as a force: an infinitely short, infinitely large force
#    cannot be integrated numerically;
# 3. **restart** the integrator from the new state.
#
# Each call of `propagate_numerical` below is one such arc: it starts a new
# integration and ends exactly at the requested time.

# %%
r_N, v_N = ad.coe2rv(r0, 0.0, 0.0, 0.0, 0.0, theta_start, mu)

r_arc_N, v_arc_N = ad.propagate_numerical(r_N, v_N, [t1], mu)             # arc 0: stop at t1
r_N, v_N = r_arc_N[0], v_arc_N[0]
v_N = ad.apply_impulse(r_N, v_N, dv1_H)                                   # add the impulse

r_arc_N, v_arc_N = ad.propagate_numerical(r_N, v_N, [t2 - t1], mu)        # arc 1: restart, stop at t2
r_N, v_N = r_arc_N[0], v_arc_N[0]
v_N = ad.apply_impulse(r_N, v_N, dv2_H)                                   # add the impulse

r_arc_N, v_arc_N = ad.propagate_numerical(r_N, v_N, [t_end - t2], mu)     # arc 2: restart
r_numerical_N, v_numerical_N = r_arc_N[0], v_arc_N[0]

difference = np.linalg.norm(r_numerical_N - r_end_N)
print("final position, numerical:", r_numerical_N, "km")
print(f"difference from the analytical solution = {difference * 1e6:.3f} mm")
assert difference < 1e-5                                                  # less than 1 cm
print("Analytical and numerical propagation agree.")

# %% [markdown]
# ## 7. Figure
#
# The three orbits (dashed) and the trajectory actually flown (solid): arc 0
# on the circle, arc 1 after the first impulse, arc 2 after the second. The
# arrows show the two impulses. The trajectory is continuous, with a corner
# at each burn point.

# %%
def orbit_points(p, e, argp, theta_from_deg, theta_to_deg):
    """Points of an orbit in the plane, between two polar angles (degrees)."""
    theta = np.radians(np.linspace(theta_from_deg, theta_to_deg, 200))
    r = ad.orbit_radius(p, e, theta - argp)
    return r * np.cos(theta), r * np.sin(theta)


fig, ax = plt.subplots(figsize=(6.5, 6.5))
for p, e, argp in ((r0, 0.0, 0.0), (p1, e1, argp1), (p2, e2, argp2)):          # the whole orbits
    ax.plot(*orbit_points(p, e, argp, 0.0, 360.0), color="gray", linestyle="--", linewidth=0.8)
ax.plot(*orbit_points(r0, 0.0, 0.0, theta_start_deg, 360.0), color="black", linewidth=2.5, label="arc 0")
ax.plot(*orbit_points(p1, e1, argp1, 0.0, theta_2_deg), color="C0", linewidth=2.5, label="arc 1")
ax.plot(*orbit_points(p2, e2, argp2, theta_2_deg, theta_end_deg), color="C3", linewidth=2.5, label="arc 2")

arrow_scale = 0.7 * r0 / max(dv1, dv2)                                          # km of arrow per km/s
for point_N, impulse_N, name in ((r_burn1_N, dv1_N, "impulse 1"), (r_burn2_N, dv2_N, "impulse 2")):
    tip = point_N[:2] + arrow_scale * impulse_N[:2]
    ax.annotate("", xy=tip, xytext=point_N[:2],
                arrowprops={"arrowstyle": "->", "color": "C1", "linewidth": 2.0})
    ax.plot(point_N[0], point_N[1], "ko")
    ax.plot(tip[0], tip[1], alpha=0.0)                                          # keeps the tip inside the figure
    ax.annotate(name, tip, color="C1")
ax.add_patch(plt.Circle((0.0, 0.0), ad.R_EARTH, color="0.8"))                   # the Earth, to scale
ax.set_aspect("equal")
ax.set_xlabel("km")
ax.set_ylabel("km")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 8. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r0": r0, "theta_start": theta_start_deg, "e1": e1, "argp1": argp1_deg, "theta_2": theta_2_deg,
        "e2": e2, "argp2": argp2_deg, "theta_end": theta_end_deg, "mu": mu}
results = {"p1": p1, "a1": a1, "p2": p2, "a2": a2, "r_burn2": r_burn2, "dv1_H": dv1_H, "dv2_H": dv2_H,
           "dv1": dv1, "dv2": dv2, "dv_total": dv_total, "t1": t1, "t2": t2, "t_end": t_end,
           "r_end_N": r_end_N, "v_end_N": v_end_N}
validation.compare("L6-6", data, results)

# %% [markdown]
# ## Now try
#
# 1. Both impulses have a radial component. Which part of each impulse
#    changes the energy of the orbit? (Example problems 6.3 and 6.9.)
# 2. Set `e2 = 0.0`: the second impulse then makes the orbit circular at the
#    distance of the second burn point. How does the total cost change?
# 3. In section 6, replace the three arcs by one single call of
#    `propagate_numerical` over the whole time, without the impulses. Where
#    does the spacecraft end up? This is the trajectory with no burns.
