"""Validation problems: worked problems that anyone can reproduce.

Each problem has a statement, the given data, and a ``solve`` function that
computes the results with the library. The expected results are stored in
``data/validation_expected.json``. The same definitions feed three things, so
they cannot drift apart:

- the document ``docs/problems.md`` (made by ``tools/make_docs.py``),
- the automatic tests (``tests/test_validation.py``),
- the lecture notebooks.

The statements and the numbers were written for this library and are not
taken from a book. The "inspired by" line names the book whose worked
examples suggested the kind of problem.

Units of the data: km, s, kg; angles are given and reported in degrees and
converted to radians inside ``solve``.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Callable

import numpy as np

from . import anomalies, conics, elements, kepler, lagrange, rotations, twobody
from .constants import G0, MU_EARTH, OMEGA_EARTH, R_EARTH, SIDEREAL_DAY
from .units import UnitSystem

# --- Lecture 6: module used by its problems ----------------------------------
from . import maneuvers  # noqa: E402

# --- Definition of a problem -------------------------------------------------

CURTIS = "Curtis, *Orbital Mechanics for Engineering Students*"
VALLADO = "Vallado, *Fundamentals of Astrodynamics and Applications*"
SCHAUB = "Schaub and Junkins, *Analytical Mechanics of Space Systems*"

DEG = np.pi / 180.0


@dataclass(frozen=True)
class Problem:
    """One validation problem."""

    id: str
    lecture: int
    title: str
    statement: str
    given: dict  # name -> (value, unit)
    solve: Callable[[dict], dict]  # {name: value} -> {name: result}
    result_units: dict  # name -> unit
    inspired_by: str
    verification: str  # how the result is checked independently
    digits: int = 7  # significant digits printed in the document
    rtol: float = 1e-9  # relative tolerance of the comparison with the stored results

    def values(self) -> dict:
        """The given data as a plain ``{name: value}`` dictionary."""
        return {name: value for name, (value, _) in self.given.items()}

    def results(self) -> dict:
        """Solve the problem with the library."""
        return self.solve(self.values())


# --- Lecture 1 ---------------------------------------------------------------

_STATE_A = {
    "r_N": ([5200.0, 4100.0, -2900.0], "km"),
    "v_N": ([-4.1, 5.9, 2.0], "km/s"),
    "mu": (MU_EARTH, "km^3/s^2"),
}


def _solve_integrals(g):
    r_N, v_N, mu = np.array(g["r_N"]), np.array(g["v_N"]), g["mu"]
    energy = twobody.specific_energy(r_N, v_N, mu)
    return {
        "h": np.linalg.norm(twobody.angular_momentum(r_N, v_N)),
        "e": np.linalg.norm(twobody.eccentricity_vector(r_N, v_N, mu)),
        "energy": energy,
        "a": -mu / (2.0 * energy),
    }


def _solve_numerical(g):
    r_N, v_N = twobody.propagate_numerical(g["r_N"], g["v_N"], [g["t"]], g["mu"])
    return {"r_N": r_N[0], "v_N": v_N[0]}


def _solve_rk4_step_size(g):
    r0_N, v0_N, mu = np.array(g["r_N"]), np.array(g["v_N"]), g["mu"]
    a, *_ = elements.rv2coe(r0_N, v0_N, mu)
    T = conics.period(a, mu)
    # After exactly one period the spacecraft must be back at the start, so
    # the distance from the initial position is the error of the integration.
    errors = {}
    for n_steps in (g["steps_coarse"], g["steps_fine"]):
        _, r_N, _ = twobody.propagate_rk4(r0_N, v0_N, T / n_steps, int(n_steps), mu)
        errors[n_steps] = np.linalg.norm(r_N[-1] - r0_N)
    coarse, fine = errors[g["steps_coarse"]], errors[g["steps_fine"]]
    return {"period": T, "error_coarse": coarse, "error_fine": fine, "ratio": coarse / fine}


def _solve_tolerance(g):
    r0_N, v0_N, mu = np.array(g["r_N"]), np.array(g["v_N"]), g["mu"]
    a, *_ = elements.rv2coe(r0_N, v0_N, mu)
    t_end = g["revolutions"] * conics.period(a, mu)
    results = {}
    for name in ("rtol_loose", "rtol_medium", "rtol_tight"):
        r_N, _ = twobody.propagate_numerical(r0_N, v0_N, [t_end], mu, rtol=g[name])
        results["error_" + name[5:]] = np.linalg.norm(r_N[0] - r0_N)
    return results


def _solve_gravity_decay(g):
    mu = g["mu"]
    radii = {"surface": g["r_surface"], "low": g["r_surface"] + g["altitude_low"], "geo": g["r_geo"]}
    gravity = {name: twobody.gravitational_acceleration(r, mu) * 1e3 for name, r in radii.items()}   # m/s^2
    return {
        "g_surface": gravity["surface"],
        "g_low": gravity["low"],
        "g_geo": gravity["geo"],
        "ratio_low": 100.0 * gravity["low"] / gravity["surface"],
        "ratio_geo": 100.0 * gravity["geo"] / gravity["surface"],
        "r_half": np.sqrt(2.0) * g["r_surface"],            # where the acceleration is half the surface value
    }


# --- Lecture 2 ---------------------------------------------------------------


def _solve_apse_speeds(g):
    r_p, r_a, mu = g["R"] + g["perigee_altitude"], g["R"] + g["apogee_altitude"], g["mu"]
    a = 0.5 * (r_p + r_a)
    return {
        "a": a,
        "e": (r_a - r_p) / (r_a + r_p),
        "v_perigee": conics.visviva_speed(r_p, a, mu),
        "v_apogee": conics.visviva_speed(r_a, a, mu),
        "period": conics.period(a, mu),
    }


def _solve_bound(g):
    r, v, gamma, mu = g["r"], g["v"], g["flight_path_angle"] * DEG, g["mu"]
    v_escape = conics.escape_speed(r, mu)
    energy = 0.5 * v**2 - mu / r
    h = r * v * np.cos(gamma)
    e = np.sqrt(1.0 + 2.0 * energy * h**2 / mu**2)
    return {
        "v_escape": v_escape,
        "energy": energy,
        "v_infinity": np.sqrt(v**2 - v_escape**2),
        "e": e,
        "orbit_type": conics.orbit_type(e),
    }


def _solve_effective_potential(g):
    p, mu, r = g["p"], g["mu"], g["r_sample"]
    h = np.sqrt(mu * p)                                      # all the orbits share this angular momentum
    results = {"h": h, "V_min": float(conics.effective_potential(p, h, mu)),
               "V_sample": float(conics.effective_potential(r, h, mu))}
    for name, e in (("circular", 0.0), ("elliptic", g["e_elliptic"]), ("parabolic", 1.0), ("hyperbolic", g["e_hyperbolic"])):
        energy = -mu * (1.0 - e**2) / (2.0 * p) + 0.0
        results["energy_" + name] = energy
        # Turning points of the radial motion: energy = V_eff, i.e. 2 energy r^2 + 2 mu r - h^2 = 0.
        if abs(energy) < 1e-12:
            turning = [h**2 / (2.0 * mu)]
        else:
            discriminant = mu**2 + 2.0 * energy * h**2
            root = np.sqrt(discriminant) if discriminant > 1e-12 * mu**2 else 0.0       # zero: the circular orbit
            turning = sorted(x for x in ((-mu + root) / (2.0 * energy), (-mu - root) / (2.0 * energy)) if x > 0.0)
        results["periapsis_" + name] = turning[0]
        if name == "elliptic":
            results["apoapsis_elliptic"] = turning[1]
        if name != "circular":
            results["radial_speed_" + name] = np.sqrt(2.0 * (energy - results["V_sample"]))   # at r_sample
    results["v_infinity_hyperbolic"] = np.sqrt(2.0 * results["energy_hyperbolic"])
    return results


# --- Lecture 3 ---------------------------------------------------------------


def _solve_geosynchronous(g):
    r = conics.synchronous_radius(g["sidereal_day"], g["mu"])
    return {"r": r, "altitude": r - g["R"], "v": conics.circular_speed(r, g["mu"])}


def _solve_canonical(g):
    r, mu = g["R"] + g["altitude"], g["mu"]
    canonical = UnitSystem.canonical(mu, r)
    period = conics.period(r, mu)
    return {
        "period": period,
        "time_unit": canonical.time,
        "period_canonical": canonical.nondim(period, "time"),
        "speed_canonical": canonical.nondim(conics.circular_speed(r, mu), "speed"),
        "mu_canonical": canonical.nondim(mu, "mu"),
    }


# --- Lecture 4 ---------------------------------------------------------------


def _solve_position_at_time(g):
    a, e, mu = g["a"], g["e"], g["mu"]
    M = conics.mean_motion(a, mu) * g["t"]
    E = anomalies.M2E(M, e)
    f = anomalies.E2f(E, e)
    return {
        "M": M / DEG,
        "E": E / DEG,
        "f": f / DEG,
        "r": conics.orbit_radius(conics.semilatus_rectum(a, e), e, f),
    }


def _solve_time_of_flight(g):
    a, e, mu = g["a"], g["e"], g["mu"]
    t1 = kepler.time_since_periapsis(g["f1"] * DEG, a, e, mu)
    t2 = kepler.time_since_periapsis(g["f2"] * DEG, a, e, mu)
    return {"t1": t1, "t2": t2, "time_of_flight": t2 - t1, "period": conics.period(a, mu)}


def _solve_three_anomalies(g):
    a, e, mu = g["a"], g["e"], g["mu"]
    period = conics.period(a, mu)
    # The three anomalies at the given fraction of the period after the periapsis.
    M = 2.0 * np.pi * g["fraction"]
    E = anomalies.M2E(M, e)
    f = anomalies.E2f(E, e)
    # The true anomaly is farthest ahead of the mean one where they grow at
    # the same rate: df/dM = (a / r)^2 sqrt(1 - e^2) = 1.
    r_max = a * (1.0 - e**2) ** 0.25
    f_max = np.arccos((conics.semilatus_rectum(a, e) / r_max - 1.0) / e)
    M_max = anomalies.f2M(f_max, e)
    return {
        "period": period,
        "M": M / DEG,
        "E": E / DEG,
        "f": f / DEG,
        "largest_difference": (f_max - M_max) / DEG,
        "f_at_largest": f_max / DEG,
        "M_at_largest": M_max / DEG,
        "time_at_largest": M_max / conics.mean_motion(a, mu),
    }


# --- Lecture 5 ---------------------------------------------------------------

_ELEMENTS_B = {
    "a": (9500.0, "km"),
    "e": (0.15, "-"),
    "i": (52.0, "deg"),
    "raan": (40.0, "deg"),
    "argp": (110.0, "deg"),
    "f": (75.0, "deg"),
    "mu": (MU_EARTH, "km^3/s^2"),
}


def _state_B(g):
    return elements.coe2rv(
        g["a"], g["e"], g["i"] * DEG, g["raan"] * DEG, g["argp"] * DEG, g["f"] * DEG, g["mu"]
    )


def _solve_coe2rv(g):
    r_N, v_N = _state_B(g)
    return {"r_N": r_N, "v_N": v_N}


def _elements_in_degrees(coe):
    a, e, i, raan, argp, f = coe
    return {"a": a, "e": e, "i": i / DEG, "raan": raan / DEG, "argp": argp / DEG, "f": f / DEG}


def _solve_rv2coe(g):
    return _elements_in_degrees(elements.rv2coe(g["r_N"], g["v_N"], g["mu"]))


def _solve_lagrange(g):
    r0_N, v0_N = _state_B(g)
    df = g["df"] * DEG
    F, G, Fdot, Gdot = lagrange.lagrange_coefficients(r0_N, v0_N, df, g["mu"])
    r_N, v_N = lagrange.propagate_lagrange(r0_N, v0_N, df, g["mu"])
    return {"F": F, "G": G, "Fdot": Fdot, "Gdot": Gdot, "r_N": r_N, "v_N": v_N}


# --- Lecture 6 ---------------------------------------------------------------


def _solve_propellant(g):
    ve = maneuvers.exhaust_velocity(g["isp"], g["g0"])
    propellant = maneuvers.propellant_mass(g["dv"], ve, g["m0"])
    return {
        "exhaust_velocity": ve,
        "mass_ratio": maneuvers.mass_ratio(g["dv"], ve),
        "propellant_mass": propellant,
        "final_mass": g["m0"] - propellant,
    }


def _solve_perigee_burn(g):
    R, mu = g["R"], g["mu"]
    r_p, r_a = R + g["perigee_altitude"], R + g["apogee_altitude"]
    a, e = 0.5 * (r_p + r_a), (r_a - r_p) / (r_a + r_p)
    r_N, v_N = elements.coe2rv(a, e, 0.0, 0.0, 0.0, 0.0, mu)  # at perigee
    v_new_N = maneuvers.apply_impulse(r_N, v_N, g["dv_H"])
    a_new, e_new, *_ = elements.rv2coe(r_N, v_new_N, mu)
    return {
        "a_new": a_new,
        "e_new": e_new,
        "perigee_altitude_new": conics.periapsis_radius(a_new, e_new) - R,
        "apogee_altitude_new": conics.apoapsis_radius(a_new, e_new) - R,
    }


def _solve_oberth(g):
    R, dv, mu = g["R"], g["dv"], g["mu"]
    r_p, r_a = R + g["perigee_altitude"], R + g["apogee_altitude"]
    a = 0.5 * (r_p + r_a)
    v_p, v_a = conics.visviva_speed(r_p, a, mu), conics.visviva_speed(r_a, a, mu)
    gain_p = maneuvers.energy_change([v_p, 0.0, 0.0], [dv, 0.0, 0.0])
    gain_a = maneuvers.energy_change([v_a, 0.0, 0.0], [dv, 0.0, 0.0])
    # The burn point stays an apse; the opposite apse of the new orbit is at 2 a_new - r.
    a_after_p = 1.0 / (2.0 / r_p - (v_p + dv) ** 2 / mu)
    a_after_a = 1.0 / (2.0 / r_a - (v_a + dv) ** 2 / mu)
    return {
        "v_perigee": v_p,
        "v_apogee": v_a,
        "energy_gain_perigee": gain_p,
        "energy_gain_apogee": gain_a,
        "ratio": gain_p / gain_a,
        "apogee_rise": 2.0 * a_after_p - r_p - r_a,
        "perigee_rise": 2.0 * a_after_a - r_a - r_p,
    }


def _solve_general_burn(g):
    r_N, v_N = _state_B(g)
    v_new_N = maneuvers.apply_impulse(r_N, v_N, g["dv_H"])
    return _elements_in_degrees(elements.rv2coe(r_N, v_new_N, g["mu"]))


def _solve_single_impulse(g):
    r1, a2, e2, mu = g["r1"], g["a2"], g["e2"], g["mu"]
    p2 = conics.semilatus_rectum(a2, e2)
    f2 = np.arccos((p2 / r1 - 1.0) / e2)  # where orbit 2 crosses the circle, outbound
    v1 = conics.circular_speed(r1, mu)
    v2 = conics.visviva_speed(r1, a2, mu)
    gamma2 = conics.flight_path_angle(e2, f2)
    return {
        "intersect": bool(maneuvers.orbits_intersect(r1, 0.0, 0.0, a2, e2, 0.0)),
        "f2": f2 / DEG,
        "v1": v1,
        "v2": v2,
        "flight_path_angle_2": gamma2 / DEG,
        "dv": maneuvers.single_impulse_dv(v1, v2, gamma2),
    }


def _solve_multiburn(g):
    r0, mu = g["r0"], g["mu"]
    e1, argp1, e2, argp2 = g["e1"], g["argp1"] * DEG, g["e2"], g["argp2"] * DEG
    theta_start, theta_2, theta_end = g["theta_start"] * DEG, g["theta_2"] * DEG, g["theta_end"] * DEG
    # Everything in the reference plane; the polar angle theta is measured from axis 1.
    # Arc 0: the circle, from theta_start to the first burn at theta = 360 deg.
    period_0 = conics.period(r0, mu)
    t1 = (2.0 * np.pi - theta_start) / (2.0 * np.pi) * period_0
    r_burn1_N, v_before1_N = elements.coe2rv(r0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
    # Arc 1: the orbit through the first burn point with the given e1 and argp1.
    f1 = (0.0 - argp1) % (2.0 * np.pi)                      # true anomaly on orbit 1 at the burn
    p1 = r0 * (1.0 + e1 * np.cos(f1))
    a1 = p1 / (1.0 - e1**2)
    _, v_after1_N = elements.coe2rv(a1, e1, 0.0, 0.0, argp1, f1, mu)
    dv1_H = change_basis_HN(r_burn1_N, v_before1_N, v_after1_N - v_before1_N)
    f1_end = (theta_2 - argp1) % (2.0 * np.pi)
    coast_1 = (kepler.time_since_periapsis(f1_end, a1, e1, mu)
               - kepler.time_since_periapsis(f1, a1, e1, mu)) % conics.period(a1, mu)
    t2 = t1 + coast_1
    r_burn2_N, v_before2_N = elements.coe2rv(a1, e1, 0.0, 0.0, argp1, f1_end, mu)
    r_burn2 = np.linalg.norm(r_burn2_N)
    # Arc 2: the orbit through the second burn point with the given e2 and argp2.
    f2 = (theta_2 - argp2) % (2.0 * np.pi)
    p2 = r_burn2 * (1.0 + e2 * np.cos(f2))
    a2 = p2 / (1.0 - e2**2)
    _, v_after2_N = elements.coe2rv(a2, e2, 0.0, 0.0, argp2, f2, mu)
    dv2_H = change_basis_HN(r_burn2_N, v_before2_N, v_after2_N - v_before2_N)
    f2_end = (theta_end - argp2) % (2.0 * np.pi)
    coast_2 = (kepler.time_since_periapsis(f2_end, a2, e2, mu)
               - kepler.time_since_periapsis(f2, a2, e2, mu)) % conics.period(a2, mu)
    r_end_N, v_end_N = elements.coe2rv(a2, e2, 0.0, 0.0, argp2, f2_end, mu)
    dv1, dv2 = np.linalg.norm(dv1_H), np.linalg.norm(dv2_H)
    return {
        "p1": p1, "a1": a1, "p2": p2, "a2": a2, "r_burn2": r_burn2,
        "dv1_H": dv1_H, "dv2_H": dv2_H, "dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2,
        "t1": t1, "t2": t2, "t_end": t2 + coast_2, "r_end_N": r_end_N, "v_end_N": v_end_N,
    }


def change_basis_HN(r_N, v_N, vector_N):
    """Components in the Hill basis of the state (r_N, v_N) of a vector given in N."""
    return rotations.change_basis(rotations.dcm_HN(r_N, v_N), vector_N)


def _solve_propellant_sequence(g):
    ve = maneuvers.exhaust_velocity(g["isp"], g["g0"])
    m0 = g["m0"]
    propellant_1 = maneuvers.propellant_mass(g["dv1"], ve, m0)
    m1 = m0 - propellant_1
    propellant_2 = maneuvers.propellant_mass(g["dv2"], ve, m1)
    return {
        "exhaust_velocity": ve,
        "propellant_1": propellant_1,
        "mass_after_1": m1,
        "propellant_2": propellant_2,
        "final_mass": m1 - propellant_2,
        "propellant_total": propellant_1 + propellant_2,
        "propellant_2_if_first": maneuvers.propellant_mass(g["dv2"], ve, m0),
    }


def _solve_finite_burn(g):
    mu = g["mu"]
    r1 = g["R"] + g["altitude"]
    r_target = conics.synchronous_radius(g["sidereal_day"], mu)
    dv = maneuvers.hohmann(r1, r_target, mu)[0]             # the impulse that reaches the target
    v_circular, period = conics.circular_speed(r1, mu), conics.period(r1, mu)
    r_mid_N, v_mid_N = elements.coe2rv(r1, 0.0, 0.0, 0.0, 0.0, 0.0, mu)   # the burn is centred here
    apoapses, losses = [], []
    for minutes in g["burn_minutes"]:
        burn_time = 60.0 * minutes
        r_on_N, v_on_N = kepler.propagate_kepler_rv(r_mid_N, v_mid_N, -0.5 * burn_time, mu)   # ignition
        r_off_N, v_off_N = maneuvers.finite_burn(r_on_N, v_on_N, dv, burn_time, mu)          # cutoff
        a, e, *_ = elements.rv2coe(r_off_N, v_off_N, mu)
        apoapses.append(conics.apoapsis_radius(a, e))
        # The impulse at r1 that would give the same orbital energy.
        dv_equivalent = np.sqrt(2.0 * (twobody.specific_energy(r_off_N, v_off_N, mu) + mu / r1)) - v_circular
        losses.append(100.0 * (1.0 - dv_equivalent / dv))
    return {
        "dv": dv,
        "r_target": r_target,
        "period_minutes": period / 60.0,
        "orbit_fraction_percent": 100.0 * 60.0 * np.asarray(g["burn_minutes"], float) / period,
        "apoapsis": np.array(apoapses),
        "dv_loss_percent": np.array(losses),
    }


def _solve_radial_impulse(g):
    R, mu = g["R"], g["mu"]
    r = R + g["altitude"]
    r_N, v_N = elements.coe2rv(r, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
    v_new_N = maneuvers.apply_impulse(r_N, v_N, [g["dv_radial"], 0.0, 0.0])
    a, e, _, _, _, f = elements.rv2coe(r_N, v_new_N, mu)
    return {
        "v_circular": conics.circular_speed(r, mu),
        "e": e,
        "f": f / DEG,
        "p": conics.semilatus_rectum(a, e),
        "perigee_altitude": conics.periapsis_radius(a, e) - R,
        "apogee_altitude": conics.apoapsis_radius(a, e) - R,
        "a_change": a - r,
    }


def _solve_reboost(g):
    R, mu = g["R"], g["mu"]
    r = R + g["altitude"]
    n, v = conics.mean_motion(r, mu), conics.circular_speed(r, mu)
    # One prograde impulse: exact change of a from the vis-viva equation.
    dv = g["dv"]
    a_new = 1.0 / (2.0 / r - (v + dv) ** 2 / mu)
    # Raising the whole orbit: a Hohmann transfer between two close circles.
    dv1, dv2, _ = maneuvers.hohmann(r, r + g["raise"], mu)
    # First-order formula against the exact one, for impulses of growing size.
    dv_table = np.asarray(g["dv_table"], float)
    exact = 1.0 / (2.0 / r - (v + dv_table) ** 2 / mu) - r
    first_order = 2.0 * dv_table / n
    return {
        "mean_motion": n,
        "period_minutes": conics.period(r, mu) / 60.0,
        "a_change_first_order": 2.0 * dv / n,
        "a_change_exact": a_new - r,
        "e_new": (a_new - r) / a_new,
        "apogee_rise": 2.0 * (a_new - r),
        "raise_dv1": dv1,
        "raise_dv2": dv2,
        "raise_dv_total": dv1 + dv2,
        "table_a_change_exact": exact,
        "table_a_change_first_order": first_order,
        "table_error_percent": 100.0 * (first_order - exact) / exact,
    }


# --- The problems ------------------------------------------------------------

_MU = (MU_EARTH, "km^3/s^2")
_R = (R_EARTH, "km")
_ELEMENT_UNITS = {"a": "km", "e": "-", "i": "deg", "raan": "deg", "argp": "deg", "f": "deg"}

PROBLEMS = (
    Problem(
        "L1-1", 1, "Integrals of motion from one state",
        "A spacecraft in orbit about the Earth is observed at one instant with the "
        "position and velocity given below. Find the magnitude of its specific "
        "angular momentum, its eccentricity, its specific energy and its "
        "semi-major axis.",
        _STATE_A, _solve_integrals,
        {"h": "km^2/s", "e": "-", "energy": "km^2/s^2", "a": "km"},
        CURTIS,
        "The same four quantities are recomputed from the state reached after a "
        "numerical integration of 3000 s: they must not change.",
    ),
    Problem(
        "L1-2", 1, "Numerical integration of the equation of motion",
        "Starting from the state of problem L1-1, integrate the two-body equation "
        "of motion numerically and find the position and velocity 3000 s later.",
        {**_STATE_A, "t": (3000.0, "s")}, _solve_numerical,
        {"r_N": "km", "v_N": "km/s"},
        CURTIS,
        "The same state is obtained analytically, by solving Kepler's equation.",
    ),
    Problem(
        "L1-3", 1, "Step size and accuracy of a fixed-step integration",
        "Starting from the state of problem L1-1, integrate the equation of "
        "motion over exactly one orbital period with the Runge-Kutta method of "
        "order 4, first with 200 equal steps and then with 400. After one "
        "period the spacecraft must be back at its starting point, so the "
        "distance between the final and the initial position is the error. "
        "Find the period, the two errors and their ratio.",
        {**_STATE_A, "steps_coarse": (200, "-"), "steps_fine": (400, "-")}, _solve_rk4_step_size,
        {"period": "s", "error_coarse": "km", "error_fine": "km", "ratio": "-"},
        VALLADO,
        "For a method of order 4 the ratio must be close to 2^4 = 16; halving "
        "the step once more must divide the error by about 16 again; and the "
        "error measured against the solution of Kepler's equation must agree.",
        digits=5, rtol=1e-5,
    ),
    Problem(
        "L1-4", 1, "Tolerance and accuracy of an adaptive integration",
        "Starting from the state of problem L1-1, integrate over exactly ten "
        "orbital periods with the adaptive integrator (method DOP853), with "
        "relative tolerances of 1e-6, 1e-9 and 1e-11 and the absolute tolerance "
        "left at the library default. Find the distance between the final and "
        "the initial position in each case. The values depend slightly on the "
        "version of SciPy: the result to retain is their order of magnitude, "
        "and that each error is ten to a few hundred times larger than the "
        "tolerance times the size of the orbit, because the tolerance bounds "
        "each step and the errors of the steps add up.",
        {**_STATE_A, "revolutions": (10, "-"), "rtol_loose": (1e-6, "-"),
         "rtol_medium": (1e-9, "-"), "rtol_tight": (1e-11, "-")},
        _solve_tolerance,
        {"error_loose": "km", "error_medium": "km", "error_tight": "km"},
        VALLADO,
        "The error must fall as the tolerance is tightened; the drift of the "
        "specific energy must fall in the same order; and the error measured "
        "against the solution of Kepler's equation must agree.",
        digits=2, rtol=0.5,
    ),
    Problem(
        "L1-5", 1, "The decay of the gravitational acceleration with the distance",
        "Find the gravitational acceleration of the Earth at its mean radius "
        "(6371 km), on a low orbit at an altitude of 500 km and at the "
        "geostationary radius (42164 km), and the last two as percentages of "
        "the first. Find the distance at which the acceleration is half the "
        "surface value. Plot the acceleration against the distance.",
        {"r_surface": (6371.0, "km"), "altitude_low": (500.0, "km"), "r_geo": (42164.0, "km"), "mu": _MU},
        _solve_gravity_decay,
        {"g_surface": "m/s^2", "g_low": "m/s^2", "g_geo": "m/s^2", "ratio_low": "%", "ratio_geo": "%", "r_half": "km"},
        CURTIS,
        "The ratios must follow the inverse-square law, (r_surface / r)^2; the "
        "acceleration must be the magnitude of the right-hand side of the "
        "equation of motion at that distance, in any direction.",
    ),
    Problem(
        "L2-1", 2, "Speeds at perigee and apogee",
        "An Earth orbit has its perigee at an altitude of 400 km and its apogee at "
        "an altitude of 4000 km. Find the semi-major axis, the eccentricity, the "
        "speeds at perigee and at apogee, and the period.",
        {"perigee_altitude": (400.0, "km"), "apogee_altitude": (4000.0, "km"), "R": _R, "mu": _MU},
        _solve_apse_speeds,
        {"a": "km", "e": "-", "v_perigee": "km/s", "v_apogee": "km/s", "period": "s"},
        CURTIS,
        "Conservation of the angular momentum (r v equal at the two apses) and "
        "of the energy.",
    ),
    Problem(
        "L2-2", 2, "Bound or not?",
        "At a distance of 9000 km from the centre of the Earth a spacecraft has a "
        "speed of 10 km/s and a flight-path angle of 20 degrees. Compare its "
        "speed with the escape speed; find its specific energy, its speed far "
        "from the Earth, its eccentricity and the type of its orbit.",
        {"r": (9000.0, "km"), "v": (10.0, "km/s"), "flight_path_angle": (20.0, "deg"), "mu": _MU},
        _solve_bound,
        {"v_escape": "km/s", "energy": "km^2/s^2", "v_infinity": "km/s", "e": "-", "orbit_type": ""},
        CURTIS,
        "The eccentricity is recomputed from the eccentricity vector of the "
        "corresponding state; the speed far away from the vis-viva equation.",
    ),
    Problem(
        "L2-3", 2, "Qualitative analysis of the orbits with the effective potential energy",
        "Four orbits about the Earth have the same angular momentum, that of a "
        "semi-latus rectum of 20000 km, and eccentricities 0, 0.6, 1 and 1.5. "
        "Plot the effective potential energy against the distance, and the four "
        "energy levels. From the graph alone find the periapsis (and apoapsis) "
        "radius of each orbit, the radial speed of the three non-circular orbits "
        "at a distance of 30000 km, and the speed far from the Earth on the "
        "hyperbola.",
        {"p": (20000.0, "km"), "e_elliptic": (0.6, "-"), "e_hyperbolic": (1.5, "-"), "r_sample": (30000.0, "km"), "mu": _MU},
        _solve_effective_potential,
        {"h": "km^2/s", "V_min": "km^2/s^2", "V_sample": "km^2/s^2",
         "energy_circular": "km^2/s^2", "energy_elliptic": "km^2/s^2", "energy_parabolic": "km^2/s^2",
         "energy_hyperbolic": "km^2/s^2", "periapsis_circular": "km", "periapsis_elliptic": "km",
         "apoapsis_elliptic": "km", "periapsis_parabolic": "km", "periapsis_hyperbolic": "km",
         "radial_speed_elliptic": "km/s", "radial_speed_parabolic": "km/s", "radial_speed_hyperbolic": "km/s",
         "v_infinity_hyperbolic": "km/s"},
        SCHAUB,
        "The turning points read on the graph must be the apses of the orbit "
        "equation, p/(1+e) and p/(1-e); the radial speed must be (mu/h) e sin f "
        "at the true anomaly of that distance; the minimum must be at r = p.",
    ),
    Problem(
        "L3-1", 3, "The geosynchronous orbit",
        "Find the radius, the altitude and the speed of the circular orbit whose "
        "period equals the sidereal rotation period of the Earth.",
        {"sidereal_day": (SIDEREAL_DAY, "s"), "R": _R, "mu": _MU},
        _solve_geosynchronous,
        {"r": "km", "altitude": "km", "v": "km/s"},
        CURTIS,
        "The speed must equal the circumference divided by the period, and the "
        "period computed from the radius must return the sidereal day.",
    ),
    Problem(
        "L3-2", 3, "A low Earth orbit in canonical units",
        "A spacecraft is on a circular orbit at an altitude of 400 km. Find its "
        "period. Then take the orbit radius as the unit of length and choose the "
        "unit of time so that the gravitational parameter equals 1: find the "
        "unit of time, and the period, the speed and the gravitational "
        "parameter in these units.",
        {"altitude": (400.0, "km"), "R": _R, "mu": _MU},
        _solve_canonical,
        {"period": "s", "time_unit": "s", "period_canonical": "TU", "speed_canonical": "DU/TU",
         "mu_canonical": "DU^3/TU^2"},
        VALLADO,
        "In canonical units the period of the reference circular orbit is 2 pi "
        "and its speed is 1, exactly.",
    ),
    Problem(
        "L4-1", 4, "Where is the spacecraft 40 minutes after perigee?",
        "An Earth orbit has a semi-major axis of 12000 km and an eccentricity of "
        "0.35. Find the mean, eccentric and true anomaly, and the distance from "
        "the centre of the Earth, 2400 s after the perigee passage.",
        {"a": (12000.0, "km"), "e": (0.35, "-"), "t": (2400.0, "s"), "mu": _MU},
        _solve_position_at_time,
        {"M": "deg", "E": "deg", "f": "deg", "r": "km"},
        CURTIS,
        "The equation of motion is integrated numerically from perigee for "
        "2400 s; the distance and the true anomaly must agree.",
    ),
    Problem(
        "L4-2", 4, "Time of flight between two points of an orbit",
        "On an orbit with a semi-major axis of 24400 km and an eccentricity of "
        "0.73, how long does the spacecraft take to go from a true anomaly of "
        "90 degrees to a true anomaly of 180 degrees?",
        {"a": (24400.0, "km"), "e": (0.73, "-"), "f1": (90.0, "deg"), "f2": (180.0, "deg"), "mu": _MU},
        _solve_time_of_flight,
        {"t1": "s", "t2": "s", "time_of_flight": "s", "period": "s"},
        CURTIS,
        "Propagating from the first point for the time found must arrive at the "
        "second point; the time to apogee must be half the period.",
    ),
    Problem(
        "L4-3", 4, "The three anomalies along an elliptical orbit",
        "On an orbit with a semi-major axis of 15000 km and an eccentricity of "
        "0.5, plot the mean, the eccentric and the true anomaly against the "
        "time, over one period. Find the three anomalies a quarter of a period "
        "after the perigee. Find the largest difference between the true and "
        "the mean anomaly, and where it occurs.",
        {"a": (15000.0, "km"), "e": (0.5, "-"), "fraction": (0.25, "-"), "mu": _MU},
        _solve_three_anomalies,
        {"period": "s", "M": "deg", "E": "deg", "f": "deg", "largest_difference": "deg",
         "f_at_largest": "deg", "M_at_largest": "deg", "time_at_largest": "s"},
        CURTIS,
        "The largest difference must agree with a search over a fine grid of "
        "mean anomalies; there the distance must be a (1 - e^2)^(1/4); the three "
        "anomalies must satisfy Kepler's equation and the half-angle relation.",
    ),
    Problem(
        "L5-1", 5, "Position and velocity from the orbital elements",
        "Find the position and the velocity, in inertial components, of a "
        "spacecraft with the classical orbital elements given below.",
        _ELEMENTS_B, _solve_coe2rv,
        {"r_N": "km", "v_N": "km/s"},
        CURTIS,
        "The distance must satisfy the orbit equation, the energy must equal "
        "-mu/(2a), and the angular momentum must make the angle i with axis 3.",
    ),
    Problem(
        "L5-2", 5, "Orbital elements from position and velocity",
        "Find the classical orbital elements of a spacecraft observed with the "
        "position and velocity given below.",
        {"r_N": ([-2700.0, 6100.0, 4300.0], "km"), "v_N": ([-5.8, -3.4, 2.2], "km/s"), "mu": _MU},
        _solve_rv2coe, _ELEMENT_UNITS,
        VALLADO,
        "Converting the elements back must return the given position and "
        "velocity.",
    ),
    Problem(
        "L5-3", 5, "A quarter of a revolution with the Lagrange coefficients",
        "Starting from the state of problem L5-1, find the four Lagrange "
        "coefficients for an increase of the true anomaly of 90 degrees, and "
        "the position and velocity there.",
        {**_ELEMENTS_B, "df": (90.0, "deg")}, _solve_lagrange,
        {"F": "-", "G": "s", "Fdot": "1/s", "Gdot": "-", "r_N": "km", "v_N": "km/s"},
        CURTIS,
        "The same state follows from the elements with the true anomaly "
        "increased by 90 degrees; the coefficients satisfy F Gdot - G Fdot = 1.",
    ),
    # --- Lecture 6 ---
    Problem(
        "L6-1", 6, "Propellant for an apogee burn",
        "A geostationary satellite has a mass of 3500 kg at separation from the "
        "launcher and must provide a velocity change of 1.47 km/s with a storable "
        "bipropellant engine of specific impulse 320 s. Find the exhaust "
        "velocity, the mass ratio, the propellant burnt and the final mass.",
        {"dv": (1.47, "km/s"), "isp": (320.0, "s"), "m0": (3500.0, "kg"), "g0": (G0, "km/s^2")},
        _solve_propellant,
        {"exhaust_velocity": "km/s", "mass_ratio": "-", "propellant_mass": "kg", "final_mass": "kg"},
        CURTIS,
        "The rocket equation applied to the initial and final mass must return "
        "the velocity change.",
    ),
    Problem(
        "L6-2", 6, "A prograde burn at perigee",
        "A spacecraft is at the perigee of an orbit with perigee altitude 300 km "
        "and apogee altitude 800 km. It fires its engine along the velocity for "
        "a velocity change of 0.1 km/s. Find the new semi-major axis, "
        "eccentricity, and perigee and apogee altitudes.",
        {"perigee_altitude": (300.0, "km"), "apogee_altitude": (800.0, "km"),
         "dv_H": ([0.0, 0.1, 0.0], "km/s"), "R": _R, "mu": _MU},
        _solve_perigee_burn,
        {"a_new": "km", "e_new": "-", "perigee_altitude_new": "km", "apogee_altitude_new": "km"},
        CURTIS,
        "The perigee must not move, and the formula for the tangential burn "
        "that moves the opposite apse must return 0.1 km/s.",
    ),
    Problem(
        "L6-3", 6, "The Oberth effect",
        "A spacecraft is on a geostationary transfer orbit with perigee altitude "
        "400 km and apogee altitude 35786 km. Compare a prograde velocity change "
        "of 0.1 km/s applied at perigee and at apogee: the specific energy "
        "gained, and how far the opposite apse rises.",
        {"perigee_altitude": (400.0, "km"), "apogee_altitude": (35786.0, "km"), "dv": (0.1, "km/s"),
         "R": _R, "mu": _MU},
        _solve_oberth,
        {"v_perigee": "km/s", "v_apogee": "km/s", "energy_gain_perigee": "km^2/s^2",
         "energy_gain_apogee": "km^2/s^2", "ratio": "-", "apogee_rise": "km", "perigee_rise": "km"},
        CURTIS,
        "Each gain is recomputed as the difference of the specific energies of "
        "the orbits before and after the burn.",
    ),
    Problem(
        "L6-4", 6, "A burn with radial and normal components",
        "The spacecraft of problem L5-1 applies a velocity change with Hill "
        "components 0.05 km/s radial, 0 transverse and 0.2 km/s normal. Find "
        "the orbital elements after the burn.",
        {**_ELEMENTS_B, "dv_H": ([0.05, 0.0, 0.2], "km/s")}, _solve_general_burn, _ELEMENT_UNITS,
        SCHAUB,
        "The position is unchanged, and the change of the magnitude of the "
        "velocity follows from the three Hill components.",
    ),
    Problem(
        "L6-5", 6, "One impulse between two intersecting orbits",
        "A spacecraft on a circular orbit of radius 7000 km must move, with one "
        "impulse, onto a coplanar orbit with semi-major axis 8400 km and "
        "eccentricity 0.2. Check that the orbits intersect; at the crossing "
        "point where the new orbit is outbound find its true anomaly, the two "
        "speeds, the flight-path angle of the new orbit and the velocity "
        "change.",
        {"r1": (7000.0, "km"), "a2": (8400.0, "km"), "e2": (0.2, "-"), "mu": _MU},
        _solve_single_impulse,
        {"intersect": "", "f2": "deg", "v1": "km/s", "v2": "km/s", "flight_path_angle_2": "deg",
         "dv": "km/s"},
        CURTIS,
        "The velocity change is recomputed as the difference of the two "
        "velocity vectors at the common point.",
    ),
    Problem(
        "L6-6", 6, "Two impulses, three Keplerian arcs",
        "A spacecraft moves in one plane; the polar angle theta is measured "
        "from a fixed axis. It starts at theta = 215 degrees on a circular "
        "orbit of radius 8000 km. At theta = 360 degrees a first impulse puts "
        "it on an orbit with eccentricity 0.4 and argument of periapsis -40 "
        "degrees. At theta = 150 degrees a second impulse puts it on an orbit "
        "with eccentricity 0.2 and argument of periapsis 100 degrees, which it "
        "follows up to theta = 290 degrees. Find the two orbits, the two "
        "impulses in Hill components (radial, transverse, normal), their "
        "magnitudes, the times of the two burns and of the end, and the final "
        "state.",
        {"r0": (8000.0, "km"), "theta_start": (215.0, "deg"),
         "e1": (0.4, "-"), "argp1": (-40.0, "deg"), "theta_2": (150.0, "deg"),
         "e2": (0.2, "-"), "argp2": (100.0, "deg"), "theta_end": (290.0, "deg"), "mu": _MU},
        _solve_multiburn,
        {"p1": "km", "a1": "km", "p2": "km", "a2": "km", "r_burn2": "km",
         "dv1_H": "km/s", "dv2_H": "km/s", "dv1": "km/s", "dv2": "km/s", "dv_total": "km/s",
         "t1": "s", "t2": "s", "t_end": "s", "r_end_N": "km", "v_end_N": "km/s"},
        CURTIS,
        "The equation of motion is integrated numerically arc by arc: stop at "
        "each burn time, add the impulse to the velocity, restart. The final "
        "state must agree.",
    ),
    Problem(
        "L6-7", 6, "Propellant for a sequence of impulses",
        "A spacecraft of 1000 kg performs two impulses with the same engine, of "
        "specific impulse 320 s: first 2.397 km/s, then 1.456 km/s (the two "
        "impulses of a Hohmann transfer from low Earth orbit to the "
        "geostationary orbit). Find the propellant of each impulse, the "
        "mass after each, the total propellant, and the propellant that the "
        "second impulse would need if it were given first.",
        {"m0": (1000.0, "kg"), "isp": (320.0, "s"), "dv1": (2.397, "km/s"), "dv2": (1.456, "km/s"),
         "g0": (G0, "km/s^2")},
        _solve_propellant_sequence,
        {"exhaust_velocity": "km/s", "propellant_1": "kg", "mass_after_1": "kg", "propellant_2": "kg",
         "final_mass": "kg", "propellant_total": "kg", "propellant_2_if_first": "kg"},
        CURTIS,
        "The total must equal the propellant of one impulse equal to the sum of "
        "the two, and must not depend on their order.",
    ),
    Problem(
        "L6-8", 6, "Finite burn or impulse?",
        "A spacecraft on a circular orbit at 400 km altitude must raise its "
        "apoapsis to the geostationary radius. The velocity change is the one "
        "of the impulsive maneuver, but it is delivered by a burn of finite "
        "duration, along the velocity, with constant thrust acceleration, "
        "centred on the point where the impulse would be given. For burn "
        "times of 1, 5, 10 and 20 minutes find the fraction of the orbit flown "
        "during the burn, the apoapsis radius reached, and the equivalent loss "
        "of velocity change (from the orbital energy reached).",
        {"altitude": (400.0, "km"), "burn_minutes": ([1.0, 5.0, 10.0, 20.0], "min"),
         "sidereal_day": (SIDEREAL_DAY, "s"), "R": _R, "mu": _MU},
        _solve_finite_burn,
        {"dv": "km/s", "r_target": "km", "period_minutes": "min", "orbit_fraction_percent": "%",
         "apoapsis": "km", "dv_loss_percent": "%"},
        VALLADO,
        "The same burns are integrated with the fixed-step Runge-Kutta method "
        "of order 4 written out independently; a very short burn must return "
        "the impulsive result.",
        digits=6, rtol=1e-7,
    ),
    Problem(
        "L6-9", 6, "A radial impulse",
        "A spacecraft is on a circular orbit at 400 km altitude. An impulse of "
        "0.1 km/s is applied radially outward. Find the new orbit: "
        "eccentricity, true anomaly of the maneuver point, semi-latus rectum, "
        "perigee and apogee altitudes, and the change of the semi-major axis.",
        {"altitude": (400.0, "km"), "dv_radial": (0.1, "km/s"), "R": _R, "mu": _MU},
        _solve_radial_impulse,
        {"v_circular": "km/s", "e": "-", "f": "deg", "p": "km", "perigee_altitude": "km",
         "apogee_altitude": "km", "a_change": "km"},
        CURTIS,
        "The angular momentum does not change, so the semi-latus rectum equals "
        "the radius of the circle; the eccentricity equals the impulse divided "
        "by the circular speed.",
    ),
    Problem(
        "L6-10", 6, "Reboost of a space station",
        "A space station is on a circular orbit at 400 km altitude. (a) For one "
        "prograde impulse of 1 m/s find the change of the semi-major axis, at "
        "first order and exactly, the new eccentricity and the rise of the "
        "apogee. (b) Find the two impulses that raise the whole circular orbit "
        "by 2 km. (c) Compare the first-order formula with the exact change of "
        "the semi-major axis for prograde impulses of 1 m/s, 10 m/s, 100 m/s, "
        "1 km/s and 2.397 km/s.",
        {"altitude": (400.0, "km"), "dv": (0.001, "km/s"), "raise": (2.0, "km"),
         "dv_table": ([0.001, 0.01, 0.1, 1.0, 2.397], "km/s"), "R": _R, "mu": _MU},
        _solve_reboost,
        {"mean_motion": "rad/s", "period_minutes": "min", "a_change_first_order": "km",
         "a_change_exact": "km", "e_new": "-", "apogee_rise": "km", "raise_dv1": "km/s",
         "raise_dv2": "km/s", "raise_dv_total": "km/s", "table_a_change_exact": "km",
         "table_a_change_first_order": "km", "table_error_percent": "%"},
        SCHAUB,
        "The impulses are applied to the state and the orbits recomputed from "
        "it: semi-major axis and eccentricity after one impulse, and a "
        "circular orbit 2 km higher after the two.",
    ),
    # --- end of the problems ---
)


def get(problem_id: str) -> Problem:
    """Return the problem with the given identifier, for example "L4-1"."""
    for problem in PROBLEMS:
        if problem.id == problem_id:
            return problem
    raise KeyError(problem_id)


def expected() -> dict:
    """The stored expected results of all problems, ``{id: {name: value}}``."""
    text = resources.files("pyastrodyn_edu").joinpath("data/validation_expected.json").read_text()
    return json.loads(text)


def compare(problem_id: str, data: dict, results: dict) -> bool:
    """Compare the results of an example with the stored results of its problem.

    Used in the last section of every example script and notebook.

    Parameters
    ----------
    problem_id : str
        Identifier of the problem, for example "L4-1".
    data : dict
        The given data used in the example, ``{name: value}``, with the names
        of ``get(problem_id).given``.
    results : dict
        The results computed in the example, ``{name: value}``, with the names
        of the stored results.

    Returns
    -------
    bool
        True if the data are the original ones and every result agrees with
        the stored one within the tolerance of the problem. One line is
        printed in every case.

    Notes
    -----
    The stored results belong to the original data. If any datum was changed,
    nothing is compared: the function says so and returns False.
    """
    problem = get(problem_id)
    for name, original in problem.values().items():
        if not np.array_equal(np.asarray(data[name]), np.asarray(original)):
            print(f"The data were changed ({name}): the stored results do not apply.")
            return False
    stored = expected()[problem_id]
    different = []
    for name, value in stored.items():
        if isinstance(value, (str, bool)):
            same = results[name] == value
        else:
            same = np.allclose(np.asarray(results[name], dtype=float), np.asarray(value),
                               rtol=problem.rtol, atol=0.0)
        if not same:
            different.append(name)
    if different:
        print("These results differ from the stored results:", ", ".join(different))
        return False
    print("The results agree with the stored results.")
    return True


def to_plain(results: dict) -> dict:
    """Convert results to plain Python numbers, lists, strings and booleans."""
    plain = {}
    for name, value in results.items():
        if isinstance(value, (str, bool)):
            plain[name] = value
        else:
            array = np.asarray(value, dtype=float)
            plain[name] = float(array) if array.ndim == 0 else array.tolist()
    return plain


__all__ = ["Problem", "PROBLEMS", "get", "expected", "compare", "to_plain"]

# --- Lecture 6: helper function of its problems ------------------------------
__all__ += ["change_basis_HN"]
