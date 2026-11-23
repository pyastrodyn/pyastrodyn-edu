# %% [markdown]
# # Example problem 1.2: Numerical integration of the equation of motion
#
# **Lecture 1, hypotheses and equation of motion.** Inspired by the worked
# examples of Curtis, *Orbital Mechanics for Engineering Students*; the
# statement and the numbers were written for this library.
#
# **Problem.** Starting from the state of Example problem 1.1, integrate the
# two-body equation of motion numerically and find the position and velocity
# 3000 s later.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km and s. Vectors are given by their components in the inertial
# basis N.

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
r0_N = np.array([5200.0, 4100.0, -2900.0])   # initial position, km (original: [5200.0, 4100.0, -2900.0])
v0_N = np.array([-4.1, 5.9, 2.0])            # initial velocity, km/s (original: [-4.1, 5.9, 2.0])
t = 3000.0                                   # time after the initial instant, s (original: 3000.0)
mu = ad.MU_EARTH                             # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The equation of motion
#
# The acceleration of the spacecraft is -mu r / |r|^3. To integrate, position
# and velocity are stacked in one array of six numbers, the state x, and the
# equation is written as x' = f(t, x): the derivative of the position is the
# velocity, and the derivative of the velocity is the acceleration. The
# function `twobody_eom` is this f.

# %%
x0_N = np.concatenate([r0_N, v0_N])          # the state: position, then velocity
xdot0_N = ad.twobody_eom(0.0, x0_N, mu)      # its derivative at the initial instant

print("state     :", x0_N)
print("derivative:", np.round(xdot0_N, 6))
print(f"acceleration magnitude = {np.linalg.norm(xdot0_N[3:]) * 1000.0:.3f} m/s^2")

# %% [markdown]
# ## 3. Solution with the library
#
# `propagate_numerical` integrates the equation of motion with an adaptive
# Runge-Kutta method (DOP853 of SciPy, relative tolerance 1e-11) and returns
# the position and velocity at each requested time. Here one time is
# requested, so the result has one row.

# %%
r_all_N, v_all_N = ad.propagate_numerical(r0_N, v0_N, [t], mu)
r_N, v_N = r_all_N[0], v_all_N[0]            # the one row

print("position:", r_N, "km")
print("velocity:", v_N, "km/s")

# %% [markdown]
# ## 4. Independent check: the analytical solution
#
# The two-body problem can also be solved without integrating: convert the
# state to orbital elements, advance the anomaly with Kepler's equation
# (Lecture 4), and convert back. `propagate_kepler_rv` does this. The two
# methods share no code.

# %%
r_kepler_N, v_kepler_N = ad.propagate_kepler_rv(r0_N, v0_N, t, mu)

difference = np.linalg.norm(r_N - r_kepler_N)          # km
print("position by Kepler's equation:", r_kepler_N, "km")
print(f"difference between the two methods = {difference * 1e6:.3f} mm")

if difference < 1e-5:                                    # less than 1 cm
    print("Numerical and analytical solution agree.")
else:
    print("The two solutions differ by more than 1 cm: the error of a numerical")
    print("integration grows with the time (see Example problem 1.4).")

# %% [markdown]
# ## 5. Figure
#
# The orbit over one period, with the initial point and the point reached
# after the time t.

# %%
a, e, *_ = ad.rv2coe(r0_N, v0_N, mu)
T = ad.period(a, mu)
times = np.linspace(0.0, T, 300)
r_orbit_N, _ = ad.propagate_numerical(r0_N, v0_N, times, mu)

fig, ax = ad.plot_orbit_3d(r_orbit_N, label="orbit")
ax.plot(*r0_N, "o", label="initial point")
ax.plot(*r_N, "s", label=f"after {t:.0f} s")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"r_N": r0_N, "v_N": v0_N, "t": t, "mu": mu}
results = {"r_N": r_N, "v_N": v_N}
validation.compare("L1-2", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `t` equal to the period `T` of section 5 (first compute it, then
#    type its value). Where is the spacecraft?
# 2. Integrate for 100 periods and look at the difference of section 4. How
#    does it grow? Example problem 1.4 studies this.
# 3. Press `F12` on `twobody_eom`: the whole physics of the problem is in
#    that one function.
