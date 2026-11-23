"""First steps with PyAstroDyn-edu.

Run this file to check that the library works on your computer:

    python examples/first_steps.py

or, in Visual Studio Code, open the file and press the Run button (the
triangle at the top right of the editor).

Units: km and s. Angles are in radians inside the library; this script
converts to degrees only to print.
"""

import numpy as np

import pyastrodyn_edu as ad

mu = ad.MU_EARTH        # gravitational parameter of the Earth, km^3/s^2
R = ad.R_EARTH          # equatorial radius of the Earth, km

print("PyAstroDyn-edu version", ad.__version__)
print()

# --- 1. From orbital elements to position and velocity, and back -------------
a, e = 9500.0, 0.15                                        # km, -
i, raan, argp, f = np.radians([52.0, 40.0, 110.0, 75.0])   # degrees -> radians

r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, mu)           # components in the inertial basis N
print("1. State from the orbital elements")
print("   position (km)   ", np.round(r_N, 3))
print("   velocity (km/s) ", np.round(v_N, 5))

a_back, e_back, *_ = ad.rv2coe(r_N, v_N, mu)
print(f"   back to elements: a = {a_back:.3f} km, e = {e_back:.5f}")
print()

# --- 2. How long is one revolution? ------------------------------------------
T = ad.period(a, mu)
print(f"2. Orbital period: {T:.1f} s = {T / 60:.1f} min")
print()

# --- 3. Where is the spacecraft a quarter of a period later? ----------------
r_later_N, v_later_N = ad.propagate_kepler_rv(r_N, v_N, T / 4.0, mu)
*_, f_later = ad.rv2coe(r_later_N, v_later_N, mu)
print("3. A quarter of a period later")
print("   position (km)   ", np.round(r_later_N, 3))
print(f"   true anomaly    {np.degrees(f):.1f} deg -> {np.degrees(f_later):.1f} deg")
print()

print("If you see three results above and no error message, the library works.")
