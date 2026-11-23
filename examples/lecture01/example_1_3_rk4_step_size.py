# %% [markdown]
# # Example problem 1.3: Step size and accuracy of a fixed-step integration
#
# **Lecture 1, hypotheses and equation of motion.** Inspired by Vallado,
# *Fundamentals of Astrodynamics and Applications*; the statement and the
# numbers were written for this library.
#
# **Problem.** Starting from the state of Example problem 1.1, integrate the
# equation of motion over exactly one orbital period with the Runge-Kutta
# method of order 4, first with 200 equal steps and then with 400. After one
# period the spacecraft must be back at its starting point, so the distance
# between the final and the initial position is the error. Find the period,
# the two errors and their ratio.
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
r0_N = np.array([5200.0, 4100.0, -2900.0])   # initial position, km (original: [5200.0, 4100.0, -2900.0])
v0_N = np.array([-4.1, 5.9, 2.0])            # initial velocity, km/s (original: [-4.1, 5.9, 2.0])
steps_coarse = 200                           # number of steps, first run (original: 200)
steps_fine = 400                             # number of steps, second run (original: 400)
mu = ad.MU_EARTH                             # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. The period
#
# From the state to the semi-major axis, and from the semi-major axis to the
# period, T = 2 pi sqrt(a^3 / mu).

# %%
a, e, *_ = ad.rv2coe(r0_N, v0_N, mu)
T = ad.period(a, mu)
print(f"a = {a:.3f} km, e = {e:.5f}, period T = {T:.3f} s")

# %% [markdown]
# ## 3. One step of the Runge-Kutta method of order 4
#
# The method advances the state x by a step h with four evaluations of the
# derivative f: at the start (k1), twice at the middle (k2, k3) and at the
# end (k4) of the step, and combines them with weights 1, 2, 2, 1:
#
# x(t + h) = x + h (k1 + 2 k2 + 2 k3 + k4) / 6.
#
# `rk4_step` does one such step. Below, the same step written out.

# %%
h = T / steps_coarse                         # the step, s
x_N = np.concatenate([r0_N, v0_N])           # the state: position, then velocity

k1 = ad.twobody_eom(0.0, x_N, mu)
k2 = ad.twobody_eom(0.0, x_N + 0.5 * h * k1, mu)
k3 = ad.twobody_eom(0.0, x_N + 0.5 * h * k2, mu)
k4 = ad.twobody_eom(0.0, x_N + h * k3, mu)
x_next_N = x_N + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

assert np.allclose(x_next_N, ad.rk4_step(x_N, h, mu))
print(f"step h = {h:.4f} s")
print("state after one step:", np.round(x_next_N, 5))

# %% [markdown]
# ## 4. Solution: one period with 200 and with 400 steps
#
# `propagate_rk4` repeats the step and returns the times, positions and
# velocities at every step. The last row is the state after one period.

# %%
errors = {}
for n_steps in (steps_coarse, steps_fine):
    times, r_N, v_N = ad.propagate_rk4(r0_N, v0_N, T / n_steps, n_steps, mu)
    errors[n_steps] = np.linalg.norm(r_N[-1] - r0_N)       # distance from the starting point, km
    print(f"{n_steps:5d} steps: error = {errors[n_steps] * 1000.0:.4f} m")

error_coarse, error_fine = errors[steps_coarse], errors[steps_fine]
ratio = error_coarse / error_fine
print(f"ratio of the two errors = {ratio:.3f}")

# %% [markdown]
# ## 5. Independent check: the order of the method
#
# For a method of order 4 the error is proportional to h^4, so halving the
# step must divide the error by about 2^4 = 16. Halve the step once more and
# look at the ratio again.

# %%
_, r_N, _ = ad.propagate_rk4(r0_N, v0_N, T / (2 * steps_fine), 2 * steps_fine, mu)
error_finer = np.linalg.norm(r_N[-1] - r0_N)
print(f"{2 * steps_fine:5d} steps: error = {error_finer * 1000.0:.4f} m, ratio = {error_fine / error_finer:.3f}")

if steps_fine == 2 * steps_coarse and steps_coarse >= 100:
    assert 12.0 < ratio < 20.0 and 12.0 < error_fine / error_finer < 20.0
    print("Both ratios are close to 16: the method is of order 4.")

# %% [markdown]
# ## 6. Figure
#
# The error against the number of steps, on logarithmic axes. A method of
# order 4 gives a straight line of slope -4.

# %%
step_numbers = np.array([50, 100, 200, 400, 800, 1600])
step_errors = []
for n_steps in step_numbers:
    _, r_N, _ = ad.propagate_rk4(r0_N, v0_N, T / n_steps, int(n_steps), mu)
    step_errors.append(np.linalg.norm(r_N[-1] - r0_N))
step_errors = np.array(step_errors)

fig, ax = plt.subplots()
ax.loglog(step_numbers, step_errors * 1000.0, "o-", label="Runge-Kutta 4")
ax.loglog(step_numbers, step_errors[0] * 1000.0 * (step_numbers[0] / step_numbers) ** 4, "--",
          label="slope -4")
ax.set_xlabel("number of steps per period")
ax.set_ylabel("position error after one period, m")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1. The stored
# results of this problem are reproducible to about five digits.

# %%
data = {"r_N": r0_N, "v_N": v0_N, "steps_coarse": steps_coarse, "steps_fine": steps_fine, "mu": mu}
results = {"period": T, "error_coarse": error_coarse, "error_fine": error_fine, "ratio": ratio}
validation.compare("L1-3", data, results)

# %% [markdown]
# ## Now try
#
# 1. How many steps per period are needed for an error of 1 mm? Read it from
#    the figure, then verify.
# 2. Set `steps_coarse = 20` and `steps_fine = 40`. Is the ratio still near
#    16? Why not?
# 3. In section 6 extend `step_numbers` up to 51200. Where does the line stop
#    being straight, and what limits the accuracy there?
