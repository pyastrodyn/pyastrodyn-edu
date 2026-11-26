# %% [markdown]
# # Example problem 6.7: Propellant for a sequence of impulses
#
# **Lecture 6, the rocket equation: propellant mass for a sequence of
# impulses.** Inspired by the worked examples of Curtis, *Orbital Mechanics
# for Engineering Students*; the statement and the numbers were written for
# this library.
#
# **Problem.** A spacecraft of 1000 kg performs two impulses with the same
# engine, of specific impulse 320 s: first 2.397 km/s, then 1.456 km/s (the
# two impulses of a Hohmann transfer from low Earth orbit to the
# geostationary orbit). Find the propellant of each
# impulse, the mass after each, the total propellant, and the propellant that
# the second impulse would need if it were given first.
#
# This is the example of the slide "Propellant mass for a sequence of
# impulses" of Lecture 6.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km, s and kg.

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
m0 = 1000.0      # mass before the first impulse, kg (original: 1000.0)
isp = 320.0      # specific impulse of the engine, s (original: 320.0)
dv1 = 2.397      # first impulse, km/s (original: 2.397)
dv2 = 1.456      # second impulse, km/s (original: 1.456)
g0 = ad.G0       # standard acceleration of gravity, km/s^2 (original: ad.G0)

# %% [markdown]
# ## 2. Solution: one impulse after the other
#
# The rocket equation is applied to each impulse, with the mass **before
# that impulse** as initial mass: the second impulse accelerates a lighter
# spacecraft, because the propellant of the first is gone.

# %%
ve = ad.exhaust_velocity(isp, g0)                    # effective exhaust velocity, km/s

propellant_1 = ad.propellant_mass(dv1, ve, m0)       # first impulse, from the full mass
mass_after_1 = m0 - propellant_1

propellant_2 = ad.propellant_mass(dv2, ve, mass_after_1)    # second impulse, from what is left
final_mass = mass_after_1 - propellant_2

propellant_total = propellant_1 + propellant_2

print(f"exhaust velocity = {ve:.4f} km/s")
print(f"impulse 1: {propellant_1:.1f} kg of {m0:.1f} kg; mass left {mass_after_1:.1f} kg")
print(f"impulse 2: {propellant_2:.1f} kg of {mass_after_1:.1f} kg; mass left {final_mass:.1f} kg")
print(f"total propellant = {propellant_total:.1f} kg, {100.0 * propellant_total / m0:.1f} % of the initial mass")

# %% [markdown]
# ## 3. The same solution, step by step
#
# For the impulse k: m_k = m_(k-1) exp(-dv_k / v_e). Multiplying the mass
# ratios of all the impulses, the exponents add up:
#
# m_N = m_0 exp(-dv_total / v_e).
#
# So the total propellant depends only on the sum of the impulses: this is
# why the sum of the velocity changes is used as the cost of a sequence of
# maneuvers.

# %%
mass_after_1_steps = m0 * np.exp(-dv1 / ve)
final_mass_steps = mass_after_1_steps * np.exp(-dv2 / ve)

dv_total = dv1 + dv2
final_mass_one_burn = m0 * np.exp(-dv_total / ve)        # as if it were one impulse

assert np.isclose(mass_after_1_steps, mass_after_1) and np.isclose(final_mass_steps, final_mass)
assert np.isclose(final_mass_one_burn, final_mass)
print(f"dv_total = {dv_total:.3f} km/s; final mass from one equation = {final_mass_one_burn:.1f} kg")
print("Two impulses cost the same propellant as one impulse equal to their sum.")

# %% [markdown]
# ## 4. Independent check: the other order
#
# Give the second impulse first. Each impulse now needs a different amount
# of propellant, but the total must be the same.

# %%
propellant_2_if_first = ad.propellant_mass(dv2, ve, m0)
propellant_1_if_second = ad.propellant_mass(dv1, ve, m0 - propellant_2_if_first)

print(f"the impulse of {dv2} km/s: {propellant_2:.1f} kg when given second, "
      f"{propellant_2_if_first:.1f} kg when given first")
print(f"total in the other order = {propellant_2_if_first + propellant_1_if_second:.1f} kg")

assert np.isclose(propellant_2_if_first + propellant_1_if_second, propellant_total)
print("The total does not depend on the order.")

# %% [markdown]
# ## 5. Figure
#
# The mass of the spacecraft against the velocity change delivered so far.
# The curve is one exponential; the two impulses are two pieces of it.

# %%
delivered = np.linspace(0.0, dv_total, 200)
fig, ax = plt.subplots()
ax.plot(delivered, m0 * np.exp(-delivered / ve))
ax.plot([0.0, dv1, dv_total], [m0, mass_after_1, final_mass], "ko")
ax.annotate("start", (0.0, m0), textcoords="offset points", xytext=(8, 0))
ax.annotate("after impulse 1", (dv1, mass_after_1), textcoords="offset points", xytext=(8, 4))
ax.annotate("after impulse 2", (dv_total, final_mass), textcoords="offset points", xytext=(-90, 8))
ax.set_xlabel("velocity change delivered, km/s")
ax.set_ylabel("mass, kg")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"m0": m0, "isp": isp, "dv1": dv1, "dv2": dv2, "g0": g0}
results = {"exhaust_velocity": ve, "propellant_1": propellant_1, "mass_after_1": mass_after_1,
           "propellant_2": propellant_2, "final_mass": final_mass, "propellant_total": propellant_total,
           "propellant_2_if_first": propellant_2_if_first}
validation.compare("L6-7", data, results)

# %% [markdown]
# ## Now try
#
# 1. The design direction: the final (dry) mass is given, 300 kg, and the
#    initial mass is asked. Compute it backwards, m_(k-1) = m_k exp(dv_k / v_e).
# 2. Use a different engine for the second impulse, with specific impulse
#    220 s. Is the total still given by the sum of the two impulses?
# 3. With `isp = 450.0`, what fraction of the initial mass reaches the final
#    orbit?
