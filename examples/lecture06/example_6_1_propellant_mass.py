# %% [markdown]
# # Example problem 6.1: Propellant for an apogee burn
#
# **Lecture 6, the rocket equation.** Inspired by the worked examples of
# Curtis, *Orbital Mechanics for Engineering Students*; the statement and the
# numbers were written for this library.
#
# **Problem.** A geostationary satellite has a mass of 3500 kg at separation
# from the launcher and must provide a velocity change of 1.47 km/s with a
# storable bipropellant engine of specific impulse 320 s. Find the exhaust
# velocity, the mass ratio, the propellant burnt and the final mass.
#
# This is the first example of the slide "Rocket equation: two examples" of
# Lecture 6; the second one is in section 5.
#
# This example exists as a notebook and as a script with the same content.
# How to use them: Step 11 of `docs/getting_started.md`.
#
# Units: km, s and kg. The standard acceleration of gravity is therefore in
# km/s^2, and the exhaust velocity comes out in km/s.

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
dv = 1.47        # velocity change, km/s (original: 1.47)
isp = 320.0      # specific impulse of the engine, s (original: 320.0)
m0 = 3500.0      # mass before the burn, kg (original: 3500.0)
g0 = ad.G0       # standard acceleration of gravity, km/s^2 (original: ad.G0)

# %% [markdown]
# ## 2. Solution with the library

# %%
ve = ad.exhaust_velocity(isp, g0)            # effective exhaust velocity, km/s
ratio = ad.mass_ratio(dv, ve)                # mass before over mass after
propellant = ad.propellant_mass(dv, ve, m0)  # propellant burnt, kg
final_mass = m0 - propellant

print(f"exhaust velocity = {ve:.6f} km/s")
print(f"mass ratio       = {ratio:.6f}")
print(f"propellant       = {propellant:.3f} kg")
print(f"final mass       = {final_mass:.3f} kg")

# %% [markdown]
# ## 3. The same solution, step by step
#
# - The specific impulse is the exhaust velocity divided by g0: v_e = Isp g0.
# - The rocket equation of Tsiolkovsky, dv = v_e ln(m0 / mf), solved for the
#   mass ratio: m0 / mf = exp(dv / v_e).
# - The propellant is the difference between the two masses.

# %%
ve_steps = isp * g0
ratio_steps = np.exp(dv / ve_steps)
final_mass_steps = m0 / ratio_steps
propellant_steps = m0 - final_mass_steps

assert np.isclose(ve_steps, ve) and np.isclose(ratio_steps, ratio) and np.isclose(propellant_steps, propellant)
print(f"propellant fraction = {propellant_steps / m0:.4f} of the initial mass")
print("Sections 2 and 3 agree.")

# %% [markdown]
# ## 4. Independent check: the rocket equation in the other direction
#
# From the initial and the final mass, the rocket equation must return the
# velocity change.

# %%
dv_check = ad.rocket_dv(ve, m0, final_mass)
print(f"velocity change from the two masses = {dv_check:.9f} km/s")
assert np.isclose(dv_check, dv)
print("The rocket equation returns the velocity change.")

# %% [markdown]
# ## 5. The second example of the slide: from the ground to low Earth orbit
#
# Reaching low Earth orbit from the ground requires about 9.4 km/s, losses
# included. Even with the best chemical propellants (specific impulse 450 s)
# most of the initial mass must be propellant: this is why launchers have
# stages.

# %%
dv_launch = 9.4                              # km/s, losses included
ve_launch = ad.exhaust_velocity(450.0, g0)
ratio_launch = ad.mass_ratio(dv_launch, ve_launch)

print(f"exhaust velocity = {ve_launch:.3f} km/s")
print(f"mass ratio = {ratio_launch:.2f}")
print(f"propellant fraction = {1.0 - 1.0 / ratio_launch:.3f}; left for everything else: {1.0 / ratio_launch:.3f}")

# %% [markdown]
# ## 6. Figure
#
# The fraction of the initial mass that is propellant, against the velocity
# change, for three specific impulses. The curve is an exponential: each
# additional km/s costs more than the previous one.

# %%
velocity_changes = np.linspace(0.0, 6.0, 200)
fig, ax = plt.subplots()
for specific_impulse in (220.0, 320.0, 450.0):
    fraction = 1.0 - 1.0 / ad.mass_ratio(velocity_changes, ad.exhaust_velocity(specific_impulse, g0))
    ax.plot(velocity_changes, fraction, label=f"Isp = {specific_impulse:.0f} s")
ax.plot(dv, propellant / m0, "ko", label="this problem")
ax.set_xlabel("velocity change, km/s")
ax.set_ylabel("propellant mass / initial mass")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Comparison with the stored results
#
# The comparison applies only to the original data of section 1.

# %%
data = {"dv": dv, "isp": isp, "m0": m0, "g0": g0}
results = {"exhaust_velocity": ve, "mass_ratio": ratio, "propellant_mass": propellant, "final_mass": final_mass}
validation.compare("L6-1", data, results)

# %% [markdown]
# ## Now try
#
# 1. Double the velocity change. Does the propellant double?
# 2. With an electric engine, `isp = 1800.0`, how much propellant is needed?
# 3. Two burns of 0.735 km/s, one after the other: is the total propellant
#    the same as for one burn of 1.47 km/s? Example problem 6.7 answers.
