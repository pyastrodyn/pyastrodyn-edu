"""Tests of the library functions.

The checks are of four kinds: round trips (a conversion followed by its
inverse), agreement between independent methods, identities that the physics
imposes, and "many states in one call equal many single calls".
"""

import inspect

import numpy as np
import pytest

import pyastrodyn_edu as ad

MU = ad.MU_EARTH
RNG = np.random.default_rng(20261122)
MODULES = ("twobody", "conics", "anomalies", "kepler", "rotations", "elements",
           "lagrange", "units", "plotting", "single_state")


def random_elliptic_elements(n):
    return (
        RNG.uniform(7000.0, 40000.0, n), RNG.uniform(0.01, 0.8, n),
        RNG.uniform(0.05, np.pi - 0.05, n), RNG.uniform(0.0, 2 * np.pi, n),
        RNG.uniform(0.0, 2 * np.pi, n), RNG.uniform(0.0, 2 * np.pi, n),
    )


def angle_difference(x, y):
    return np.abs(np.angle(np.exp(1j * (np.asarray(x) - np.asarray(y)))))


# --- Documentation rule ------------------------------------------------------


def test_every_public_function_cites_a_book_and_a_lecture():
    for module_name in MODULES:
        module = getattr(ad, module_name)
        for name in module.__all__:
            doc = inspect.getdoc(getattr(module, name)) or ""
            assert "References\n----------" in doc, f"{module_name}.{name}: no References block"
            block = doc.split("References\n----------\n", 1)[1].split("\n\n", 1)[0]
            assert "Lecture " in block, f"{module_name}.{name}: no lecture cited"
            assert "*" in block, f"{module_name}.{name}: no book cited"


# --- Rotations ---------------------------------------------------------------


def test_elementary_rotations_are_proper_and_passive():
    angles = RNG.uniform(-7.0, 7.0, 20)
    for rot in (ad.rot1, ad.rot2, ad.rot3):
        C = rot(angles)
        assert C.shape == (20, 3, 3)
        assert np.allclose(C @ ad.invert_dcm(C), np.eye(3), atol=1e-14)
        assert np.allclose(np.linalg.det(C), 1.0)
        assert np.allclose(rot(angles[3]), C[3])
    # Passive: turning the basis by +theta about axis 3 makes a vector fixed
    # along the old axis 1 appear at the angle -theta.
    theta = 0.3
    assert np.allclose(ad.rot3(theta) @ [1.0, 0.0, 0.0], [np.cos(theta), -np.sin(theta), 0.0])
    assert np.allclose(ad.rot1(theta) @ [0.0, 1.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)])
    assert np.allclose(ad.rot2(theta) @ [0.0, 0.0, 1.0], [-np.sin(theta), 0.0, np.cos(theta)])


def test_dcm_NP_columns_are_the_perifocal_axes():
    i, raan, argp = 0.7, 1.9, 4.1
    r_N, v_N = ad.coe2rv(9000.0, 0.3, i, raan, argp, 0.0, MU)  # at periapsis
    C_NP = ad.dcm_NP(i, raan, argp)
    assert np.allclose(C_NP[:, 0], r_N / np.linalg.norm(r_N))
    h_N = np.cross(r_N, v_N)
    assert np.allclose(C_NP[:, 2], h_N / np.linalg.norm(h_N))


def test_dcm_HN_resolves_position_and_angular_momentum():
    r_N, v_N = ad.coe2rv(*random_elliptic_elements(30), MU)
    C_HN = ad.dcm_HN(r_N, v_N)
    assert C_HN.shape == (30, 3, 3)
    r_H = ad.change_basis(C_HN, r_N)
    assert np.allclose(r_H[:, 0], np.linalg.norm(r_N, axis=1))
    assert np.allclose(r_H[:, 1:], 0.0, atol=1e-9)
    h_H = ad.change_basis(C_HN, np.cross(r_N, v_N))
    assert np.allclose(h_H[:, :2], 0.0, atol=1e-6)
    assert np.all(ad.change_basis(C_HN, v_N)[:, 1] > 0.0)  # transverse velocity is positive
    assert np.allclose(ad.dcm_HN(r_N[5], v_N[5]), C_HN[5])
    assert np.allclose(ad.dcm_EN(0.4), ad.rot3(0.4))


# --- Conics ------------------------------------------------------------------


def test_conic_relations():
    a, e = 12000.0, 0.4
    r_p, r_a = ad.periapsis_radius(a, e), ad.apoapsis_radius(a, e)
    p = ad.semilatus_rectum(a, e)
    assert np.isclose(ad.orbit_radius(p, e, 0.0), r_p)
    assert np.isclose(ad.orbit_radius(p, e, np.pi), r_a)
    # Angular momentum at the two apses.
    assert np.isclose(r_p * ad.visviva_speed(r_p, a, MU), r_a * ad.visviva_speed(r_a, a, MU))
    assert np.isclose(ad.visviva_speed(a, a, MU), ad.circular_speed(a, MU))
    assert np.isclose(ad.escape_speed(a, MU), np.sqrt(2.0) * ad.circular_speed(a, MU))
    assert np.isclose(ad.period(a, MU) * ad.mean_motion(a, MU), 2 * np.pi)
    assert np.isclose(ad.period(ad.synchronous_radius(86164.0, MU), MU), 86164.0)
    assert ad.apoapsis_radius(-20000.0, 1.5) == np.inf
    with pytest.raises(ValueError):
        ad.period(-1.0, MU)
    # The minimum of the effective potential is at the circular radius h^2/mu.
    h = np.sqrt(MU * p)
    r = np.linspace(0.5 * p, 2.0 * p, 20001)
    assert np.isclose(r[np.argmin(ad.effective_potential(r, h, MU))], p, rtol=1e-4)


def test_orbit_type():
    assert list(ad.orbit_type([0.0, 0.5, 1.0, 1.3])) == ["circular", "elliptical", "parabolic", "hyperbolic"]
    assert ad.orbit_type(0.2) == "elliptical"


def test_flight_path_angle_two_ways():
    a, e, i, raan, argp, f = random_elliptic_elements(40)
    r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, MU)
    assert np.allclose(ad.flight_path_angle(e, f), ad.flight_path_angle_rv(r_N, v_N))
    assert np.isclose(ad.flight_path_angle(0.3, 0.0), 0.0)


# --- Anomalies and Kepler's equation -----------------------------------------


@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99, 0.999])
def test_anomaly_round_trips_over_several_revolutions(e):
    f = np.linspace(-4 * np.pi, 4 * np.pi, 801)
    E = ad.f2E(f, e)
    assert np.allclose(ad.E2f(E, e), f, atol=1e-10)
    M = ad.E2M(E, e)
    assert np.allclose(ad.M2E(M, e), E, atol=1e-9)
    assert np.allclose(ad.M2f(ad.f2M(f, e), e), f, atol=1e-8)
    assert np.all(np.diff(E) > 0) and np.all(np.diff(M) > 0)  # continuous, increasing
    # Residual of Kepler's equation.
    E_solved = ad.M2E(M, e)
    assert np.allclose(E_solved - e * np.sin(E_solved), M, atol=1e-11)


def test_f2E_agrees_with_the_half_angle_formula():
    e = 0.6
    f = np.linspace(-3.0, 3.0, 61)
    E_half_angle = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2))
    assert np.allclose(ad.f2E(f, e), E_half_angle)


def test_anomalies_reject_non_elliptic_eccentricity():
    with pytest.raises(ValueError):
        ad.f2E(1.0, 1.2)


def test_time_since_periapsis_and_propagate_kepler():
    a, e = 15000.0, 0.45
    T = ad.period(a, MU)
    assert np.isclose(ad.time_since_periapsis(np.pi, a, e, MU), T / 2)
    assert np.isclose(ad.time_since_periapsis(0.0, a, e, MU), 0.0)
    f0 = 1.1
    assert np.isclose(ad.propagate_kepler(a, e, f0, T, MU), f0 + 2 * np.pi)
    assert np.isclose(ad.propagate_kepler(a, e, ad.propagate_kepler(a, e, f0, 1234.0, MU), -1234.0, MU), f0)


# --- Elements ----------------------------------------------------------------


def test_coe_rv_round_trip_many_states_at_once():
    coe = random_elliptic_elements(300)
    r_N, v_N = ad.coe2rv(*coe, MU)
    assert r_N.shape == (300, 3)
    back = ad.rv2coe(r_N, v_N, MU)
    assert np.allclose(back[0], coe[0], rtol=1e-10)
    assert np.allclose(back[1], coe[1], atol=1e-11)
    for k in (2, 3, 4, 5):
        assert np.all(angle_difference(back[k], coe[k]) < 1e-8)


def test_many_states_equal_single_calls():
    coe = random_elliptic_elements(8)
    r_N, v_N = ad.coe2rv(*coe, MU)
    many = ad.rv2coe(r_N, v_N, MU)
    for k in range(8):
        r_k, v_k = ad.coe2rv(*(c[k] for c in coe), MU)
        assert np.array_equal(r_k, r_N[k]) and np.array_equal(v_k, v_N[k])
        single = ad.rv2coe(r_N[k], v_N[k], MU)
        assert np.allclose(single, [m[k] for m in many], rtol=1e-13, atol=1e-13)


def test_one_orbit_with_many_true_anomalies():
    f = np.linspace(0.0, 6.0, 25)
    r_N, v_N = ad.coe2rv(9000.0, 0.2, 0.8, 1.0, 2.0, f, MU)
    assert r_N.shape == (25, 3)
    assert np.allclose(ad.specific_energy(r_N, v_N, MU), -MU / 18000.0)


def test_hyperbolic_round_trip():
    coe = (-25000.0, 1.4, 0.6, 2.0, 1.0, 0.9)
    assert np.allclose(ad.rv2coe(*ad.coe2rv(*coe, MU), MU), coe)
    with pytest.raises(ValueError):
        ad.coe2rv(8000.0, 1.2, 0.0, 0.0, 0.0, 0.0, MU)


SINGULAR_CASES = {
    "circular inclined": (7000.0, 0.0, 0.9, 1.3, 0.0, 2.2),
    "elliptical equatorial": (9000.0, 0.3, 0.0, 0.0, 1.7, 0.6),
    "circular equatorial": (7000.0, 0.0, 0.0, 0.0, 0.0, 4.0),
    "elliptical retrograde equatorial": (9000.0, 0.3, np.pi, 0.0, 1.7, 0.6),
    "circular retrograde equatorial": (7000.0, 0.0, np.pi, 0.0, 0.0, 4.0),
    "circular polar": (7000.0, 0.0, np.pi / 2, 0.4, 0.0, 5.0),
}


@pytest.mark.parametrize("name", SINGULAR_CASES)
def test_singular_cases_are_exact_inverses(name):
    coe = SINGULAR_CASES[name]
    back = ad.rv2coe(*ad.coe2rv(*coe, MU), MU)
    assert np.isclose(back[0], coe[0]) and np.isclose(back[1], coe[1], atol=1e-12)
    for k in (2, 3, 4, 5):
        assert angle_difference(back[k], coe[k]) < 1e-7, (name, k)


def test_mixed_regular_and_singular_states_in_one_call():
    cases = list(SINGULAR_CASES.values()) + [(8000.0, 0.2, 0.9, 1.1, 2.0, 0.7)]
    coe = tuple(np.array(column) for column in zip(*cases))
    r_N, v_N = ad.coe2rv(*coe, MU)
    back = ad.rv2coe(r_N, v_N, MU)
    r_back, v_back = ad.coe2rv(*back, MU)
    assert np.allclose(r_back, r_N, atol=1e-6) and np.allclose(v_back, v_N, atol=1e-9)
    for k, case in enumerate(cases):
        single = ad.rv2coe(r_N[k], v_N[k], MU)
        assert np.allclose(single, [b[k] for b in back], atol=1e-12)


def test_equinoctial_round_trip_gives_the_same_state():
    cases = list(SINGULAR_CASES.values())[:3] + [(8000.0, 0.2, 0.9, 1.1, 2.0, 0.7)]
    coe = tuple(np.array(column) for column in zip(*cases))
    equinoctial = ad.coe2equinoctial(*coe)
    r_N, v_N = ad.coe2rv(*coe, MU)
    r_back, v_back = ad.coe2rv(*ad.equinoctial2coe(*equinoctial), MU)
    assert np.allclose(r_back, r_N, atol=1e-6) and np.allclose(v_back, v_N, atol=1e-9)
    regular = (8000.0, 0.2, 0.9, 1.1, 2.0, 0.7)
    assert np.allclose(ad.equinoctial2coe(*ad.coe2equinoctial(*regular)), regular)


# --- Two-body problem and propagation ----------------------------------------


def test_integrals_of_motion():
    a, e, i, raan, argp, f = random_elliptic_elements(50)
    r_N, v_N = ad.coe2rv(a, e, i, raan, argp, f, MU)
    assert np.allclose(ad.specific_energy(r_N, v_N, MU), -MU / (2 * a))
    assert np.allclose(np.linalg.norm(ad.eccentricity_vector(r_N, v_N, MU), axis=1), e)
    h_N = ad.angular_momentum(r_N, v_N)
    assert np.allclose(np.linalg.norm(h_N, axis=1), np.sqrt(MU * a * (1 - e**2)))
    assert np.allclose(h_N[:, 2] / np.linalg.norm(h_N, axis=1), np.cos(i))


def test_equation_of_motion_shapes_and_value():
    x_N = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    xdot_N = ad.twobody_eom(0.0, x_N, MU)
    assert np.allclose(xdot_N, [0.0, 7.5, 0.0, -MU / 7000.0**2, 0.0, 0.0])
    assert ad.twobody_eom(0.0, np.tile(x_N, (4, 1)), MU).shape == (4, 6)


def test_three_propagators_agree():
    coe = (11000.0, 0.35, 0.9, 0.5, 1.2, 0.4)
    r0_N, v0_N = ad.coe2rv(*coe, MU)
    T = ad.period(coe[0], MU)
    times = np.linspace(0.0, 1.5 * T, 13)
    r_num, v_num = ad.propagate_numerical(r0_N, v0_N, times, MU)
    r_kep, v_kep = ad.propagate_kepler_rv(r0_N, v0_N, times, MU)
    assert np.allclose(r_num, r_kep, atol=1e-4)  # 0.1 m after 1.5 revolutions
    assert np.allclose(v_num, v_kep, atol=1e-7)
    # Lagrange coefficients, driven by the true anomaly reached at each time.
    df = ad.propagate_kepler(coe[0], coe[1], coe[5], times, MU) - coe[5]
    r_lag, v_lag = ad.propagate_lagrange(r0_N, v0_N, df, MU)
    assert np.allclose(r_lag, r_kep, atol=1e-7) and np.allclose(v_lag, v_kep, atol=1e-10)
    F, G, Fdot, Gdot = ad.lagrange_coefficients(r0_N, v0_N, df, MU)
    assert np.allclose(F * Gdot - G * Fdot, 1.0)


def test_numerical_propagation_returns_after_one_period():
    r0_N, v0_N = ad.coe2rv(8000.0, 0.1, 1.0, 2.0, 3.0, 4.0, MU)
    r_N, v_N = ad.propagate_numerical(r0_N, v0_N, [ad.period(8000.0, MU)], MU)
    assert np.allclose(r_N[0], r0_N, atol=1e-5) and np.allclose(v_N[0], v0_N, atol=1e-8)


def test_lagrange_is_valid_on_a_hyperbola_and_at_half_a_turn():
    r0_N, v0_N = ad.coe2rv(-25000.0, 1.4, 0.6, 2.0, 1.0, -0.5, MU)
    r_N, v_N = ad.propagate_lagrange(r0_N, v0_N, 1.2, MU)
    assert np.allclose((r_N, v_N), ad.coe2rv(-25000.0, 1.4, 0.6, 2.0, 1.0, 0.7, MU))
    r0_N, v0_N = ad.coe2rv(9000.0, 0.2, 0.3, 0.0, 0.0, 0.3, MU)
    r_N, v_N = ad.propagate_lagrange(r0_N, v0_N, np.pi, MU)
    assert np.allclose((r_N, v_N), ad.coe2rv(9000.0, 0.2, 0.3, 0.0, 0.0, 0.3 + np.pi, MU))


# --- Units and scaling -------------------------------------------------------


def test_canonical_units():
    canonical = ad.UnitSystem.canonical(MU, 7000.0)
    assert np.isclose(canonical.nondim(MU, "mu"), 1.0)
    assert np.isclose(canonical.nondim(ad.circular_speed(7000.0, MU), "speed"), 1.0)
    assert np.isclose(canonical.nondim(ad.period(7000.0, MU), "time"), 2 * np.pi)
    assert np.isclose(canonical.dim(canonical.nondim(123.4, "acceleration"), "acceleration"), 123.4)
    with pytest.raises(ValueError):
        canonical.unit("mass")


def test_results_do_not_depend_on_the_unit_system():
    canonical = ad.UnitSystem.canonical(MU, 6678.0)
    # A full state conversion and a numerical propagation in canonical units.
    r_N, v_N = ad.coe2rv(9000.0, 0.2, 0.8, 1.0, 2.0, 0.9, MU)
    r_c, v_c = canonical.nondim(r_N, "length"), canonical.nondim(v_N, "speed")
    assert np.allclose(ad.rv2coe(r_c, v_c, 1.0)[1:], ad.rv2coe(r_N, v_N, MU)[1:])
    t = 2000.0
    r_end, _ = ad.propagate_numerical(r_N, v_N, [t], MU)
    r_end_c, _ = ad.propagate_numerical(r_c, v_c, [canonical.nondim(t, "time")], 1.0)
    assert np.allclose(canonical.dim(r_end_c, "length"), r_end, atol=1e-5)


# --- Purity ------------------------------------------------------------------


def test_inputs_are_not_modified():
    r_N = np.array([7000.0, 100.0, -50.0])
    v_N = np.array([0.1, 7.4, 1.0])
    copies = r_N.copy(), v_N.copy()
    ad.rv2coe(r_N, v_N, MU)
    ad.propagate_lagrange(r_N, v_N, 1.0, MU)
    ad.propagate_kepler_rv(r_N, v_N, 100.0, MU)
    ad.dcm_HN(r_N, v_N)
    assert np.array_equal(r_N, copies[0]) and np.array_equal(v_N, copies[1])


# --- change_basis and invert_dcm (passive convention) ------------------------


def test_change_basis_is_the_matrix_product_and_does_not_move_the_vector():
    C_BA = ad.rot3(0.4) @ ad.rot1(1.1)
    v_A = np.array([1.0, -2.0, 0.5])
    v_B = ad.change_basis(C_BA, v_A)
    assert np.allclose(v_B, C_BA @ v_A)
    assert np.isclose(np.linalg.norm(v_B), np.linalg.norm(v_A))       # same vector, same length
    assert np.allclose(ad.change_basis(ad.invert_dcm(C_BA), v_B), v_A)  # and back
    # One DCM with many vectors; many DCMs with many vectors.
    V_A = RNG.normal(size=(5, 3))
    assert np.allclose(ad.change_basis(C_BA, V_A), V_A @ C_BA.T)
    C = ad.rot2(RNG.uniform(0, 6, 5))
    many = ad.change_basis(C, V_A)
    assert all(np.allclose(many[k], C[k] @ V_A[k]) for k in range(5))


# --- Single-state twins ------------------------------------------------------


def test_single_state_twins_agree_on_regular_states():
    coe = random_elliptic_elements(60)
    r_N, v_N = ad.coe2rv(*coe, MU)
    many = ad.rv2coe(r_N, v_N, MU)
    for k in range(60):
        r_k, v_k = ad.coe2rv_single(*(c[k] for c in coe), MU)
        assert np.allclose(r_k, r_N[k], rtol=1e-12, atol=1e-9)
        assert np.allclose(v_k, v_N[k], rtol=1e-12, atol=1e-12)
        single = ad.rv2coe_single(r_N[k], v_N[k], MU)
        assert np.allclose(single, [m[k] for m in many], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", SINGULAR_CASES)
def test_single_state_twins_agree_on_singular_states(name):
    coe = SINGULAR_CASES[name]
    r_N, v_N = ad.coe2rv(*coe, MU)
    r_s, v_s = ad.coe2rv_single(*coe, MU)
    assert np.allclose(r_s, r_N, atol=1e-9) and np.allclose(v_s, v_N, atol=1e-12)
    assert np.allclose(ad.rv2coe_single(r_N, v_N, MU), ad.rv2coe(r_N, v_N, MU), atol=1e-12)


def test_single_state_twin_handles_the_hyperbola():
    coe = (-25000.0, 1.4, 0.6, 2.0, 1.0, 0.9)
    assert np.allclose(ad.rv2coe_single(*ad.coe2rv_single(*coe, MU), MU), coe)


@pytest.mark.parametrize("e", [0.0, 0.3, 0.79, 0.8, 0.95, 0.999])
def test_kepler_solver_twin_agrees(e):
    M = np.linspace(-9.0, 9.0, 73)
    E_many = ad.M2E(M, e)
    E_single = np.array([ad.M2E_single(float(m), e) for m in M])
    assert np.allclose(E_single, E_many, rtol=0, atol=1e-11)
    with pytest.raises(ValueError):
        ad.M2E_single(1.0, 1.0)


# --- Integrators -------------------------------------------------------------


def _reference_case():
    coe = (9000.0, 0.2, 0.9, 0.5, 1.2, 0.4)
    r0_N, v0_N = ad.coe2rv(*coe, MU)
    return r0_N, v0_N, ad.period(coe[0], MU)


def test_rk4_error_falls_by_about_sixteen_when_the_step_is_halved():
    r0_N, v0_N, T = _reference_case()
    r_exact, _ = ad.propagate_kepler_rv(r0_N, v0_N, T, MU)
    errors = []
    for n_steps in (200, 400, 800):
        t, r_N, v_N = ad.propagate_rk4(r0_N, v0_N, T / n_steps, n_steps, MU)
        assert t.shape == (n_steps + 1,) and r_N.shape == (n_steps + 1, 3)
        assert np.isclose(t[-1], T) and np.array_equal(r_N[0], r0_N)
        errors.append(np.linalg.norm(r_N[-1] - r_exact))
    assert 13.0 < errors[0] / errors[1] < 19.0     # order 4: a factor 2^4 = 16
    assert 13.0 < errors[1] / errors[2] < 19.0


def test_rk4_step_many_states_equal_single_steps():
    r_N, v_N = ad.coe2rv(*random_elliptic_elements(5), MU)
    x_N = np.hstack((r_N, v_N))
    many = ad.rk4_step(x_N, 30.0, MU)
    assert all(np.allclose(many[k], ad.rk4_step(x_N[k], 30.0, MU)) for k in range(5))


def test_integrator_options_and_the_effect_of_the_tolerance():
    r0_N, v0_N, T = _reference_case()
    r_exact, _ = ad.propagate_kepler_rv(r0_N, v0_N, T, MU)

    def error(**options):
        r_N, _ = ad.propagate_numerical(r0_N, v0_N, [T], MU, **options)
        return np.linalg.norm(r_N[0] - r_exact)

    assert error() < 1e-4                                   # defaults: below 0.1 m
    assert error(method="RK45", rtol=1e-10) < 1e-2          # another method works
    assert error(atol=1e-9) < 1e-3                          # explicit absolute tolerance
    # SciPy's own default tolerances are far too loose for an orbit.
    assert error(rtol=1e-3, atol=1e-6) > 1000 * error()
