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


# --- Lecture 6 ---------------------------------------------------------------


def test_independent_L6_1():
    g, res = solved("L6-1")
    assert np.isclose(res["exhaust_velocity"] * np.log(g["m0"] / res["final_mass"]), g["dv"])
    assert np.isclose(res["propellant_mass"] + res["final_mass"], g["m0"])
    assert np.isclose(res["mass_ratio"], g["m0"] / res["final_mass"])


def test_independent_L6_2():
    g, res = solved("L6-2")
    r_p = g["R"] + g["perigee_altitude"]
    assert np.isclose(res["perigee_altitude_new"], g["perigee_altitude"])
    dv = ad.apse_change_dv(r_p, g["R"] + g["apogee_altitude"], g["R"] + res["apogee_altitude_new"], g["mu"])
    assert np.isclose(dv, g["dv_H"][1])
    assert res["apogee_altitude_new"] > g["apogee_altitude"]


def test_independent_L6_3():
    g, res = solved("L6-3")
    mu, dv, R = g["mu"], g["dv"], g["R"]
    r_p, r_a = R + g["perigee_altitude"], R + g["apogee_altitude"]
    a = 0.5 * (r_p + r_a)
    e = (r_a - r_p) / (r_a + r_p)
    # From states: apply the burn at each apse and look at the new orbit.
    for where, f, rise in (("perigee", 0.0, "apogee_rise"), ("apogee", np.pi, "perigee_rise")):
        r_N, v_N = ad.coe2rv(a, e, 0.0, 0.0, 0.0, f, mu)
        v_new_N = ad.apply_impulse(r_N, v_N, [0.0, dv, 0.0])
        gain = ad.specific_energy(r_N, v_new_N, mu) + mu / (2 * a)
        assert np.isclose(res[f"energy_gain_{where}"], gain)
        a_new, e_new, *_ = ad.rv2coe(r_N, v_new_N, mu)
        opposite_old = r_a if where == "perigee" else r_p
        opposite_new = a_new * (1 + e_new) if where == "perigee" else a_new * (1 - e_new)
        if where == "apogee":  # after raising the perigee, the burn point is still the apogee
            assert np.isclose(a_new * (1 + e_new), r_a)
        assert np.isclose(res[rise], opposite_new - opposite_old, rtol=1e-7)
    assert res["ratio"] > 1.0


def test_independent_L6_4():
    g, res = solved("L6-4")
    r0_N, v0_N = ad.coe2rv(g["a"], g["e"], g["i"] * DEG, g["raan"] * DEG, g["argp"] * DEG,
                           g["f"] * DEG, g["mu"])
    r_N, v_N = ad.coe2rv(res["a"], res["e"], res["i"] * DEG, res["raan"] * DEG, res["argp"] * DEG,
                         res["f"] * DEG, g["mu"])
    assert np.allclose(r_N, r0_N, atol=1e-7)  # the position does not change
    dv_N = v_N - v0_N
    assert np.isclose(np.linalg.norm(dv_N), np.linalg.norm(g["dv_H"]))
    radial_N = r0_N / np.linalg.norm(r0_N)
    normal_N = np.cross(r0_N, v0_N) / np.linalg.norm(np.cross(r0_N, v0_N))
    assert np.isclose(dv_N @ radial_N, g["dv_H"][0]) and np.isclose(dv_N @ normal_N, g["dv_H"][2])


def test_independent_L6_5():
    g, res = solved("L6-5")
    f2 = res["f2"] * DEG
    r2_N, v2_N = ad.coe2rv(g["a2"], g["e2"], 0, 0, 0, f2, g["mu"])
    _, v1_N = ad.coe2rv(g["r1"], 0, 0, 0, 0, f2, g["mu"])
    assert res["intersect"] is True
    assert np.isclose(np.linalg.norm(r2_N), g["r1"])
    assert np.isclose(np.linalg.norm(v2_N - v1_N), res["dv"])
    assert np.isclose(np.linalg.norm(v2_N), res["v2"]) and np.isclose(np.linalg.norm(v1_N), res["v1"])


def test_independent_L6_6():
    g, res = solved("L6-6")
    mu = g["mu"]
    # Numerical propagation arc by arc: stop at each burn, add the impulse, restart.
    r_N, v_N = ad.coe2rv(g["r0"], 0.0, 0.0, 0.0, 0.0, g["theta_start"] * DEG, mu)
    r_N, v_N = (x[0] for x in ad.propagate_numerical(r_N, v_N, [res["t1"]], mu))
    assert np.allclose(r_N, [g["r0"], 0.0, 0.0], atol=1e-5)
    v_N = ad.apply_impulse(r_N, v_N, res["dv1_H"])
    r_N, v_N = (x[0] for x in ad.propagate_numerical(r_N, v_N, [res["t2"] - res["t1"]], mu))
    assert np.isclose(np.linalg.norm(r_N), res["r_burn2"])
    assert np.isclose(np.degrees(np.arctan2(r_N[1], r_N[0])), g["theta_2"])
    v_N = ad.apply_impulse(r_N, v_N, res["dv2_H"])
    a, e, _, _, argp, _ = ad.rv2coe(r_N, v_N, mu)
    assert np.isclose(e, g["e2"]) and np.isclose(a, res["a2"])
    r_N, v_N = (x[0] for x in ad.propagate_numerical(r_N, v_N, [res["t_end"] - res["t2"]], mu))
    assert np.allclose(r_N, res["r_end_N"], atol=1e-4)
    assert np.allclose(v_N, res["v_end_N"], atol=1e-8)


def test_independent_L6_7():
    g, res = solved("L6-7")
    ve = g["isp"] * g["g0"]
    one_burn = g["m0"] * (1.0 - np.exp(-(g["dv1"] + g["dv2"]) / ve))
    assert np.isclose(res["propellant_total"], one_burn)
    # The other order: second impulse first.
    m1 = g["m0"] - res["propellant_2_if_first"]
    other_order = res["propellant_2_if_first"] + m1 * (1.0 - np.exp(-g["dv1"] / ve))
    assert np.isclose(res["propellant_total"], other_order)
    assert res["propellant_2"] < res["propellant_2_if_first"]


def test_independent_L6_8():
    g, res = solved("L6-8")
    mu, r1 = g["mu"], g["R"] + g["altitude"]

    def derivative(x, acceleration):
        r, v = x[:3], x[3:]
        return np.concatenate([v, -mu * r / np.linalg.norm(r) ** 3 + acceleration * v / np.linalg.norm(v)])

    # The 5-minute burn with a Runge-Kutta 4 written here, 3000 steps.
    burn_time = 300.0
    acceleration = res["dv"] / burn_time
    r_mid_N, v_mid_N = np.array([r1, 0.0, 0.0]), np.array([0.0, np.sqrt(mu / r1), 0.0])
    r_N, v_N = ad.propagate_kepler_rv(r_mid_N, v_mid_N, -0.5 * burn_time, mu)
    x, h = np.concatenate([r_N, v_N]), burn_time / 3000
    for _ in range(3000):
        k1 = derivative(x, acceleration)
        k2 = derivative(x + 0.5 * h * k1, acceleration)
        k3 = derivative(x + 0.5 * h * k2, acceleration)
        k4 = derivative(x + h * k3, acceleration)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    a, e, *_ = ad.rv2coe(x[:3], x[3:], mu)
    assert np.isclose(a * (1 + e), res["apoapsis"][1], rtol=1e-8)
    # A very short burn is an impulse.
    r_N, v_N = ad.finite_burn(r_mid_N, v_mid_N, res["dv"], 1e-3, mu)
    a, e, *_ = ad.rv2coe(r_N, v_N, mu)
    assert np.isclose(a * (1 + e), res["r_target"], rtol=1e-6)
    # Longer burns reach lower and lose more.
    assert np.all(np.diff(res["apoapsis"]) < 0) and np.all(np.diff(res["dv_loss_percent"]) > 0)


def test_independent_L6_9():
    g, res = solved("L6-9")
    r = g["R"] + g["altitude"]
    v = np.sqrt(g["mu"] / r)
    assert np.isclose(res["p"], r)                       # the angular momentum did not change
    assert np.isclose(res["e"], g["dv_radial"] / v)
    assert np.isclose(res["f"], 90.0)
    p, e = r, g["dv_radial"] / v
    assert np.isclose(res["perigee_altitude"], p / (1 + e) - g["R"])
    assert np.isclose(res["apogee_altitude"], p / (1 - e) - g["R"])
    assert np.isclose(res["a_change"], p / (1 - e**2) - r)


def test_independent_L6_10():
    g, res = solved("L6-10")
    mu, r = g["mu"], g["R"] + g["altitude"]
    r_N, v_N = ad.coe2rv(r, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
    # (a) one impulse, from the state
    a, e, *_ = ad.rv2coe(r_N, ad.apply_impulse(r_N, v_N, [0.0, g["dv"], 0.0]), mu)
    assert np.isclose(res["a_change_exact"], a - r, rtol=1e-7) and np.isclose(res["e_new"], e, rtol=1e-6)
    assert np.isclose(res["apogee_rise"], a * (1 + e) - r, rtol=1e-6)
    assert abs(res["a_change_first_order"] / res["a_change_exact"] - 1) < 1e-3
    # (b) two impulses half a revolution apart
    v_N = ad.apply_impulse(r_N, v_N, [0.0, res["raise_dv1"], 0.0])
    a1, *_ = ad.rv2coe(r_N, v_N, mu)
    r_N, v_N = ad.propagate_kepler_rv(r_N, v_N, 0.5 * ad.period(a1, mu), mu)
    v_N = ad.apply_impulse(r_N, v_N, [0.0, res["raise_dv2"], 0.0])
    a2, e2, *_ = ad.rv2coe(r_N, v_N, mu)
    assert np.isclose(a2, r + g["raise"]) and e2 < 1e-9
    assert np.isclose(res["raise_dv_total"], res["mean_motion"] * g["raise"] / 2, rtol=1e-3)
    # (c) the first-order formula always underestimates, and more for larger impulses
    assert np.all(res["table_error_percent"] < 0) and np.all(np.diff(res["table_error_percent"]) < 0)


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
