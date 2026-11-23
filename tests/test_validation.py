"""Tests of the validation problems.

Two kinds of test:

1. Regression: the library still returns the stored expected results.
2. Independent check: each problem is verified by a second route that does not
   repeat the computation of its ``solve`` function (a different method, a
   conservation law, or a reconstruction from states).
"""

import numpy as np
import pytest

import pyastrodyn_edu as ad
from pyastrodyn_edu import validation

DEG = np.pi / 180.0
EXPECTED = validation.expected()
IDS = [p.id for p in validation.PROBLEMS]


def solved(problem_id):
    problem = validation.get(problem_id)
    return problem.values(), problem.results()


@pytest.mark.parametrize("problem_id", IDS)
def test_library_reproduces_the_stored_results(problem_id):
    problem = validation.get(problem_id)
    results = validation.to_plain(problem.results())
    assert results.keys() == EXPECTED[problem_id].keys()
    for name, value in results.items():
        if isinstance(value, (str, bool)):
            assert value == EXPECTED[problem_id][name]
        else:
            # Each problem states how closely its results are reproducible.
            assert np.allclose(value, EXPECTED[problem_id][name], rtol=problem.rtol, atol=1e-9), name


def test_every_problem_has_units_for_every_result_and_an_independent_check():
    checks = {name[len("test_independent_"):].replace("_", "-") for name in globals()
              if name.startswith("test_independent_")}
    for p in validation.PROBLEMS:
        assert set(p.result_units) == set(p.results()), p.id
        assert p.id in checks, f"{p.id} has no independent check"


# --- Independent checks, one per problem -------------------------------------


def test_independent_L1_1():
    g, res = solved("L1-1")
    r_N, v_N = ad.propagate_numerical(g["r_N"], g["v_N"], [3000.0], g["mu"])
    h = np.linalg.norm(np.cross(r_N[0], v_N[0]))
    energy = 0.5 * v_N[0] @ v_N[0] - g["mu"] / np.linalg.norm(r_N[0])
    assert np.isclose(h, res["h"], rtol=1e-10) and np.isclose(energy, res["energy"], rtol=1e-10)
    # Eccentricity from energy and angular momentum, not from the eccentricity vector.
    assert np.isclose(np.sqrt(1 + 2 * energy * h**2 / g["mu"] ** 2), res["e"], rtol=1e-9)
    assert res["a"] > 0 and res["e"] < 1


def test_independent_L1_2():
    g, res = solved("L1-2")
    r_N, v_N = ad.propagate_kepler_rv(g["r_N"], g["v_N"], g["t"], g["mu"])
    assert np.allclose(res["r_N"], r_N, atol=1e-5)  # 1 cm
    assert np.allclose(res["v_N"], v_N, atol=1e-8)


def test_independent_L1_3():
    g, res = solved("L1-3")
    r0_N, v0_N, mu, T = np.array(g["r_N"]), np.array(g["v_N"]), g["mu"], res["period"]
    assert 13.0 < res["ratio"] < 19.0                       # order 4: about 2^4
    # Halving the step once more divides the error by about 16 again.
    n = 2 * g["steps_fine"]
    _, r_N, _ = ad.propagate_rk4(r0_N, v0_N, T / n, n, mu)
    assert 13.0 < res["error_fine"] / np.linalg.norm(r_N[-1] - r0_N) < 19.0
    # Same error when measured against the solution of Kepler's equation.
    r_exact_N, _ = ad.propagate_kepler_rv(r0_N, v0_N, T, mu)
    _, r_N, _ = ad.propagate_rk4(r0_N, v0_N, T / g["steps_fine"], g["steps_fine"], mu)
    assert np.isclose(np.linalg.norm(r_N[-1] - r_exact_N), res["error_fine"], rtol=1e-4)
    # The period itself, from the mean motion.
    a = ad.rv2coe(r0_N, v0_N, mu)[0]
    assert np.isclose(T * ad.mean_motion(a, mu), 2 * np.pi)


def test_independent_L1_4():
    g, res = solved("L1-4")
    r0_N, v0_N, mu = np.array(g["r_N"]), np.array(g["v_N"]), g["mu"]
    a = ad.rv2coe(r0_N, v0_N, mu)[0]
    t_end = g["revolutions"] * ad.period(a, mu)
    assert res["error_loose"] > 100 * res["error_medium"] > 100 * 10 * res["error_tight"] > 0
    # The error is much larger than the tolerance times the size of the orbit:
    # the tolerance bounds each step, and the errors of the steps add up.
    for name in ("loose", "medium", "tight"):
        assert res["error_" + name] > 10 * g["rtol_" + name] * a
    energy0 = ad.specific_energy(r0_N, v0_N, mu)
    drifts = []
    for name in ("loose", "medium", "tight"):
        r_N, v_N = ad.propagate_numerical(r0_N, v0_N, [t_end], mu, rtol=g["rtol_" + name])
        drifts.append(abs(ad.specific_energy(r_N[0], v_N[0], mu) / energy0 - 1))
        r_exact_N, _ = ad.propagate_kepler_rv(r0_N, v0_N, t_end, mu)
        assert np.isclose(np.linalg.norm(r_N[0] - r_exact_N), res["error_" + name], rtol=1e-2)
    assert drifts[0] > drifts[1] > drifts[2]


def test_independent_L1_5():
    g, res = solved("L1-5")
    R, mu = g["r_surface"], g["mu"]
    assert np.isclose(res["ratio_low"], 100 * (R / (R + g["altitude_low"])) ** 2)
    assert np.isclose(res["ratio_geo"], 100 * (R / g["r_geo"]) ** 2)
    direction = np.array([0.36, -0.48, 0.8])                                             # a unit vector
    for name, r in (("g_surface", R), ("g_low", R + g["altitude_low"]), ("g_geo", g["r_geo"])):
        xdot_N = ad.twobody_eom(0.0, np.concatenate((r * direction, [0.0, 0.0, 0.0])), mu)
        assert np.isclose(np.linalg.norm(xdot_N[3:]) * 1e3, res[name])
    assert np.isclose(ad.gravitational_acceleration(res["r_half"], mu) * 1e3, 0.5 * res["g_surface"])
    assert 9.80 < res["g_surface"] < 9.83


def test_independent_L2_1():
    g, res = solved("L2-1")
    r_p, r_a = g["R"] + g["perigee_altitude"], g["R"] + g["apogee_altitude"]
    assert np.isclose(r_p * res["v_perigee"], r_a * res["v_apogee"])
    assert np.isclose(res["v_perigee"] ** 2 / 2 - g["mu"] / r_p, res["v_apogee"] ** 2 / 2 - g["mu"] / r_a)
    assert np.isclose(res["a"] * (1 - res["e"]), r_p)
    # Period by numerical integration of half an orbit from perigee to apogee.
    r_N, _ = ad.propagate_numerical([r_p, 0, 0], [0, res["v_perigee"], 0], [res["period"] / 2], g["mu"])
    assert np.allclose(r_N[0], [-r_a, 0, 0], atol=1e-4)


def test_independent_L2_2():
    g, res = solved("L2-2")
    gamma = g["flight_path_angle"] * DEG
    r_N = np.array([g["r"], 0.0, 0.0])
    v_N = g["v"] * np.array([np.sin(gamma), np.cos(gamma), 0.0])
    a, e, *_ = ad.rv2coe(r_N, v_N, g["mu"])
    assert np.isclose(e, res["e"]) and a < 0 and res["orbit_type"] == "hyperbolic"
    assert np.isclose(res["v_infinity"], np.sqrt(-g["mu"] / a))
    assert g["v"] > res["v_escape"] and res["energy"] > 0


def test_independent_L2_3():
    g, res = solved("L2-3")
    p, mu, r = g["p"], g["mu"], g["r_sample"]
    h = np.sqrt(mu * p)
    assert np.isclose(res["V_min"], -mu / (2 * p)) and np.isclose(res["energy_circular"], res["V_min"])
    grid = np.linspace(0.5 * p, 2.0 * p, 300001)
    assert np.isclose(grid[np.argmin(ad.effective_potential(grid, h, mu))], p, rtol=1e-5)   # the minimum is at r = p
    for name, e in (("circular", 0.0), ("elliptic", g["e_elliptic"]), ("parabolic", 1.0), ("hyperbolic", g["e_hyperbolic"])):
        assert np.isclose(res["periapsis_" + name], p / (1 + e))
        if name == "circular":
            continue
        # The state at the sample distance, from the orbit equation; its radial speed and its energy.
        f = np.arccos((p / r - 1) / e)
        radial_speed = mu / h * e * np.sin(f)
        assert np.isclose(res["radial_speed_" + name], radial_speed)
        speed2 = radial_speed**2 + (h / r) ** 2
        assert np.isclose(0.5 * speed2 - mu / r, res["energy_" + name], atol=1e-9)
    assert np.isclose(res["apoapsis_elliptic"], p / (1 - g["e_elliptic"]))
    assert np.isclose(res["v_infinity_hyperbolic"], mu / h * np.sqrt(g["e_hyperbolic"] ** 2 - 1))
    assert res["periapsis_hyperbolic"] > ad.R_EARTH                                      # every orbit clears the Earth


def test_independent_L3_1():
    g, res = solved("L3-1")
    assert np.isclose(res["v"], 2 * np.pi * res["r"] / g["sidereal_day"])
    assert np.isclose(ad.period(res["r"], g["mu"]), g["sidereal_day"])
    # Gravity supplies the centripetal acceleration.
    assert np.isclose(g["mu"] / res["r"] ** 2, res["v"] ** 2 / res["r"])
    assert 35700.0 < res["altitude"] < 35900.0


def test_independent_L3_2():
    _, res = solved("L3-2")
    assert np.isclose(res["period_canonical"], 2 * np.pi)
    assert np.isclose(res["speed_canonical"], 1.0) and np.isclose(res["mu_canonical"], 1.0)
    assert np.isclose(res["period"], 2 * np.pi * res["time_unit"])


def test_independent_L4_1():
    g, res = solved("L4-1")
    a, e, mu = g["a"], g["e"], g["mu"]
    r_p = a * (1 - e)
    v_p = np.sqrt(mu * (2 / r_p - 1 / a))
    r_N, _ = ad.propagate_numerical([r_p, 0, 0], [0, v_p, 0], [g["t"]], mu)
    assert np.isclose(np.linalg.norm(r_N[0]), res["r"], rtol=1e-9)
    assert np.isclose(np.arctan2(r_N[0, 1], r_N[0, 0]) / DEG, res["f"], rtol=1e-9)
    E = res["E"] * DEG
    assert np.isclose(E - e * np.sin(E), res["M"] * DEG)  # Kepler's equation
    assert np.isclose(a * (1 - e * np.cos(E)), res["r"])    # distance from E


def test_independent_L4_2():
    g, res = solved("L4-2")
    a, e, mu = g["a"], g["e"], g["mu"]
    assert np.isclose(res["t2"], res["period"] / 2)
    f_end = ad.propagate_kepler(a, e, g["f1"] * DEG, res["time_of_flight"], mu)
    assert np.isclose(f_end / DEG, g["f2"])
    # Numerical integration between the two points.
    r1_N, v1_N = ad.coe2rv(a, e, 0, 0, 0, g["f1"] * DEG, mu)
    r_N, _ = ad.propagate_numerical(r1_N, v1_N, [res["time_of_flight"]], mu)
    assert np.allclose(r_N[0], [-a * (1 + e), 0, 0], atol=1e-3)


def test_independent_L4_3():
    g, res = solved("L4-3")
    a, e = g["a"], g["e"]
    M, E, f = res["M"] * DEG, res["E"] * DEG, res["f"] * DEG
    assert np.isclose(res["M"], 360.0 * g["fraction"])
    assert np.isclose(E - e * np.sin(E), M)                                              # Kepler's equation
    assert np.isclose(np.tan(f / 2), np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))         # half-angle relation
    assert M < E < f                                                                     # first half of the orbit
    # Search of the largest difference on a fine grid, with a numerical integration for the true anomaly.
    grid = np.linspace(0.0, np.pi, 200001)
    difference = ad.M2f(grid, e) - grid
    k = np.argmax(difference)
    assert np.isclose(difference[k] / DEG, res["largest_difference"], rtol=1e-8)
    assert np.isclose(grid[k] / DEG, res["M_at_largest"], atol=2e-3)
    r = a * (1 - e**2) / (1 + e * np.cos(res["f_at_largest"] * DEG))
    assert np.isclose(r, a * (1 - e**2) ** 0.25)
    r_p = a * (1 - e)
    r_N, _ = ad.propagate_numerical([r_p, 0, 0], [0, np.sqrt(g["mu"] * (2 / r_p - 1 / a)), 0], [res["time_at_largest"]], g["mu"])
    assert np.isclose(np.arctan2(r_N[0, 1], r_N[0, 0]) / DEG, res["f_at_largest"], rtol=1e-8)


def test_independent_L5_1():
    g, res = solved("L5-1")
    r_N, v_N, mu = np.array(res["r_N"]), np.array(res["v_N"]), g["mu"]
    p = g["a"] * (1 - g["e"] ** 2)
    assert np.isclose(np.linalg.norm(r_N), p / (1 + g["e"] * np.cos(g["f"] * DEG)))
    assert np.isclose(v_N @ v_N / 2 - mu / np.linalg.norm(r_N), -mu / (2 * g["a"]))
    h_N = np.cross(r_N, v_N)
    assert np.isclose(np.linalg.norm(h_N), np.sqrt(mu * p))
    assert np.isclose(np.arccos(h_N[2] / np.linalg.norm(h_N)) / DEG, g["i"])
    node_N = np.cross([0, 0, 1], h_N)
    assert np.isclose(np.arctan2(node_N[1], node_N[0]) / DEG, g["raan"])
    # Height above the equatorial plane from the argument of latitude.
    arglat = (g["argp"] + g["f"]) * DEG
    assert np.isclose(r_N[2], np.linalg.norm(r_N) * np.sin(arglat) * np.sin(g["i"] * DEG))


def test_independent_L5_2():
    g, res = solved("L5-2")
    r_N, v_N = ad.coe2rv(res["a"], res["e"], res["i"] * DEG, res["raan"] * DEG,
                         res["argp"] * DEG, res["f"] * DEG, g["mu"])
    assert np.allclose(r_N, g["r_N"], atol=1e-8) and np.allclose(v_N, g["v_N"], atol=1e-11)


def test_independent_L5_3():
    g, res = solved("L5-3")
    r_N, v_N = ad.coe2rv(g["a"], g["e"], g["i"] * DEG, g["raan"] * DEG, g["argp"] * DEG,
                         (g["f"] + g["df"]) * DEG, g["mu"])
    assert np.allclose(res["r_N"], r_N, atol=1e-8) and np.allclose(res["v_N"], v_N, atol=1e-11)
    assert np.isclose(res["F"] * res["Gdot"] - res["G"] * res["Fdot"], 1.0)


# --- Checks on all the problems ----------------------------------------------


def test_every_orbit_of_the_problems_stays_above_the_atmosphere():
    """A validation problem must describe an orbit that can exist."""
    mu, R = ad.MU_EARTH, ad.R_EARTH
    perigee_altitudes = {}
    for problem_id in ("L1-1", "L5-2"):
        g = validation.get(problem_id).values()
        a, e, *_ = ad.rv2coe(g["r_N"], g["v_N"], mu)
        perigee_altitudes[problem_id] = a * (1 - e) - R
    for problem_id, (a_name, e_name) in {"L4-1": ("a", "e"), "L4-2": ("a", "e"), "L4-3": ("a", "e"),
                                         "L5-1": ("a", "e"), "L6-5": ("a2", "e2")}.items():
        if problem_id not in IDS:                            # a problem of a lecture that is not in this release
            continue
        g = validation.get(problem_id).values()
        perigee_altitudes[problem_id] = g[a_name] * (1 - g[e_name]) - R
    g = validation.get("L2-2").values()
    gamma = np.radians(g["flight_path_angle"])
    a, e, *_ = ad.rv2coe([g["r"], 0, 0], g["v"] * np.array([np.sin(gamma), np.cos(gamma), 0]), mu)
    perigee_altitudes["L2-2"] = a * (1 - e) - R
    for problem_id, altitude in perigee_altitudes.items():
        assert altitude > 150.0, (problem_id, altitude)
