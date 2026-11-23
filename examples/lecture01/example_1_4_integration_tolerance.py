# %% [markdown]
# # Example problem 1.4: Tolerance and accuracy of an adaptive integration
#
# **Lecture 1, hypotheses and equation of motion.** Inspired by Vallado,
# *Fundamentals of Astrodynamics and Applications*; the statement and the
# numbers were written for this library.
#
# **Problem.** Starting from the state of Example problem 1.1, integrate over
# exactly ten orbital periods with the adaptive integrator (method DOP853),
# with relative tolerances of 1e-6, 1e-9 and 1e-11 and the absolute tolerance
# left at the library default. Find the distance between the final and the
# initial position in each case.
#
# The values depend slightly on the version of SciPy: the result to retain is
# their order of magnitude, and that each error is ten to a few hundred times
# larger than the tolerance times the size of the orbit, because the
# tolerance bounds each step and the errors of the steps add up.
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
revolutions = 10                             # number of periods (original: 10)
rtol_loose = 1e-6                            # relative tolerances (original: 1e-6, 1e-9, 1e-11)
rtol_medium = 1e-9
rtol_tight = 1e-11
mu = ad.MU_EARTH                             # gravitational parameter, km^3/s^2 (original: ad.MU_EARTH)

# %% [markdown]
# ## 2. What an adaptive integrator does
#
# A fixed-step method (Example problem 1.3) uses the step it is given. An
# adaptive method chooses each step by itself: it estimates the error of the
# step, and shortens or lengthens the step to keep that estimate below
# `atol + rtol * |x|` for every component x of the state. The user gives the
# tolerances, not the step. The tolerance bounds the error of one step, not
# the error at the end.

# %%
a, e, *_ = ad.rv2coe(r0_N, v0_N, mu)
T = ad.period(a, mu)
t_end = revolutions * T
print(f"a = {a:.3f} km, period = {T:.3f} s, integration time = {t_end:.1f} s")

# %% [markdown]
# ## 3. Solution: ten periods with three tolerances
#
# After a whole number of periods the spacecraft must be back at its starting
# point: the distance from it is the error.

# %%
errors = {}
for name, rtol in (("loose", rtol_loose), ("medium", rtol_medium), ("tight", rtol_tight)):
    r_N, v_N = ad.propagate_numerical(r0_N, v0_N, [t_end], mu, rtol=rtol)
    errors[name] = np.linalg.norm(r_N[0] - r0_N)           # km
    print(f"rtol = {rtol:.0e}: error = {errors[name] * 1000.0:12.5f} m,"
          f"   error / (rtol * a) = {errors[name] / (rtol * a):.0f}")

# %% [markdown]
# ## 4. Independent check: the drift of the energy
#
# The specific energy is an integral of motion: any change of it along the
# numerical solution is an error. It must fall in the same order as the
# position error.

# %%
energy0 = ad.specific_energy(r0_N, v0_N, mu)
drifts = {}
for name, rtol in (("loose", rtol_loose), ("medium", rtol_medium), ("tight", rtol_tight)):
    r_N, v_N = ad.propagate_numerical(r0_N, v0_N, [t_end], mu, rtol=rtol)
    drifts[name] = abs(ad.specific_energy(r_N[0], v_N[0], mu) - energy0) / abs(energy0)
    print(f"rtol = {rtol:.0e}: relative drift of the energy = {drifts[name]:.2e}")

if rtol_loose > rtol_medium > rtol_tight:
    assert errors["loose"] > errors["medium"] > errors["tight"]
    assert drifts["loose"] > drifts["medium"] > drifts["tight"]
    print("Error and energy drift both fall as the tolerance is tightened.")

# %% [markdown]
# ## 5. Figure
#
# The position error after the ten periods against the relative tolerance.

# %%
tolerances = 10.0 ** np.arange(-12, -4.5, 1.0)
tolerance_errors = []
for rtol in tolerances:
    r_N, _ = ad.propagate_numerical(r0_N, v0_N, [t_end], mu, rtol=rtol)
    tolerance_errors.append(np.linalg.norm(r_N[0] - r0_N))

fig, ax = plt.subplots()
ax.loglog(tolerances, np.array(tolerance_errors) * 1000.0, "o-")
ax.set_xlabel("relative tolerance rtol")
ax.set_ylabel(f"position error after {revolutions} periods, m")
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1. The stored
# results of this problem are compared loosely (within 50 percent), because
# they depend on the version of SciPy.

# %%
data = {"r_N": r0_N, "v_N": v0_N, "revolutions": revolutions, "rtol_loose": rtol_loose,
        "rtol_medium": rtol_medium, "rtol_tight": rtol_tight, "mu": mu}
results = {"error_loose": errors["loose"], "error_medium": errors["medium"], "error_tight": errors["tight"]}
validation.compare("L1-4", data, results)

# %% [markdown]
# ## Now try
#
# 1. Set `revolutions = 100`. By what factor do the errors grow? Is the
#    growth proportional to the time?
# 2. In section 3 add the argument `method="RK45"` to `propagate_numerical`.
#    Which method is more accurate at the same tolerance? (The last section
#    then reports a difference: the stored results are for DOP853.)
# 3. Which tolerance would you choose to know the position to 1 m after
#    ten periods?
