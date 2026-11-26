"""Maneuvers 1: rocket equation, impulsive burns, coplanar transfers.

Any consistent unit system; angles in radians. Every scalar argument may be
one value or an array of n values; vectors have shape (3,) or (n, 3).

Sign convention for transfers: a positive velocity change is along the
velocity (prograde), a negative one against it (retrograde). The cost of a
maneuver is the sum of the absolute values.
"""

import numpy as np

from ._util import dot, out
from .conics import circular_speed, visviva_speed
from .rotations import change_basis, dcm_HN, invert_dcm

# --- Rocket equation ---------------------------------------------------------


def exhaust_velocity(isp, g0):
    """Effective exhaust velocity from the specific impulse, v_e = Isp g0.

    Parameters
    ----------
    isp : specific impulse, in seconds
    g0 : standard acceleration of gravity, in the chosen units
        (``constants.G0`` is in km/s^2)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, the rocket equation.
    """
    return np.asarray(isp, float) * np.asarray(g0, float)


def rocket_dv(ve, m0, mf):
    """Velocity change from the Tsiolkovsky rocket equation, dv = v_e ln(m0/mf).

    Parameters
    ----------
    ve : effective exhaust velocity
    m0 : mass before the burn
    mf : mass after the burn

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, the rocket equation.

    Notes
    -----
    Ideal rocket: no gravity or drag losses, constant exhaust velocity.
    """
    return np.asarray(ve, float) * np.log(np.asarray(m0, float) / np.asarray(mf, float))


def mass_ratio(dv, ve):
    """Initial-to-final mass ratio for a velocity change, m0/mf = exp(dv / v_e).

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, the rocket equation.
    """
    return np.exp(np.abs(np.asarray(dv, float)) / np.asarray(ve, float))


def propellant_mass(dv, ve, m0):
    """Propellant mass burnt to obtain a velocity change, starting from mass m0.

    Returns
    -------
    m0 (1 - exp(-|dv| / v_e))

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, the rocket equation.
    """
    return np.asarray(m0, float) * (1.0 - 1.0 / mass_ratio(dv, ve))


# --- Impulsive approximation -------------------------------------------------


def apply_impulse(r_N, v_N, dv_H):
    """Velocity just after an impulsive burn given in Hill components.

    In the impulsive approximation the position does not change during the
    burn and the velocity jumps by the velocity change.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
        Position and velocity just before the burn, components in N.
    dv_H : ndarray, shape (3,) or (n, 3)
        Velocity change, components in the Hill basis of the orbit before the
        burn: radial, transverse, normal.

    Returns
    -------
    v_N : ndarray, shape (3,) or (n, 3)
        Velocity just after the burn, components in N.

    References
    ----------
    Schaub and Junkins, *Analytical Mechanics of Space Systems*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 6, the impulsive approximation.
    """
    # Step 1: the Hill basis of the orbit BEFORE the burn, as a DCM.
    C_HN = dcm_HN(r_N, v_N)                      # to H, from N
    C_NH = invert_dcm(C_HN)                      # to N, from H

    # Step 2: describe the velocity change in N (same vector, other basis).
    dv_N = change_basis(C_NH, dv_H)

    # Step 3: the velocity jumps; the position is unchanged.
    return np.asarray(v_N, dtype=float) + dv_N


def energy_change(v_N, dv_N):
    """Change of specific energy caused by an impulsive velocity change.

    Parameters
    ----------
    v_N : ndarray, shape (3,) or (n, 3)
        Velocity just before the burn.
    dv_N : ndarray, shape (3,) or (n, 3)
        Velocity change, components in the same basis.

    Returns
    -------
    v . dv + |dv|^2 / 2

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, where to burn: the Oberth effect.

    Notes
    -----
    The first term grows with the speed at the burn point: the same velocity
    change adds more energy where the spacecraft is fastest.
    """
    return dot(v_N, dv_N) + 0.5 * dot(dv_N, dv_N)


# --- Single-impulse maneuvers ------------------------------------------------


def apse_change_dv(r_burn, r_opposite_old, r_opposite_new, mu):
    """Tangential velocity change at one apse that moves the opposite apse.

    Parameters
    ----------
    r_burn : radius of the apse where the burn takes place (it does not change)
    r_opposite_old : radius of the opposite apse before the burn
    r_opposite_new : radius of the opposite apse after the burn
    mu : gravitational parameter

    Returns
    -------
    Signed velocity change: positive (prograde) raises the opposite apse.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, raising and lowering the apses.
    """
    r_burn = np.asarray(r_burn, dtype=float)
    # The major axis joins the two apses, so  2a = r_burn + r_opposite.
    a_old = 0.5 * (r_burn + np.asarray(r_opposite_old, dtype=float))
    a_new = 0.5 * (r_burn + np.asarray(r_opposite_new, dtype=float))
    # At an apse the velocity is purely transverse, so a tangential burn
    # changes only its magnitude: the difference of the two vis-viva speeds
    # at the same radius.
    speed_old = visviva_speed(r_burn, a_old, mu)
    speed_new = visviva_speed(r_burn, a_new, mu)
    return speed_new - speed_old


def single_impulse_dv(v1, v2, delta_gamma):
    """Magnitude of the velocity change between two coplanar orbits at a common point.

    Parameters
    ----------
    v1, v2 : speeds on the two orbits at the common point
    delta_gamma : difference of the two flight-path angles there, radians

    Returns
    -------
    sqrt(v1^2 + v2^2 - 2 v1 v2 cos(delta_gamma))   (the velocity triangle)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, general single impulse: the velocity triangle.
    """
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    return np.sqrt(v1**2 + v2**2 - 2.0 * v1 * v2 * np.cos(delta_gamma))


def orbits_intersect(a1, e1, argp1, a2, e2, argp2, tol: float = 1e-12):
    """Whether two coplanar elliptical orbits have a common point.

    A single impulse can move a spacecraft from one orbit to another only at a
    point that belongs to both.

    Parameters
    ----------
    a1, e1, argp1 : semi-major axis, eccentricity and argument of periapsis of
        the first orbit
    a2, e2, argp2 : the same for the second orbit
    tol : relative tolerance, so that tangent orbits count as intersecting

    Returns
    -------
    bool, or an array of bool

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 6, single-impulse maneuvers: which orbits can be reached.

    Notes
    -----
    Both orbits must lie in the same plane and be ellipses (e < 1). Equating
    the two orbit equations at the same angle theta gives
    ``A cos(theta) + B sin(theta) = C``, which has a solution if and only if
    ``|C| <= sqrt(A^2 + B^2)``.
    """
    a1, e1, argp1, a2, e2, argp2 = (
        np.asarray(x, float) for x in (a1, e1, argp1, a2, e2, argp2)
    )
    if np.any(e1 >= 1.0) or np.any(e2 >= 1.0):
        raise ValueError("orbits_intersect is written for ellipses: e < 1.")
    # Step 1: semi-latus rectum of each orbit.
    p1, p2 = a1 * (1.0 - e1**2), a2 * (1.0 - e2**2)

    # Step 2: at a common point, at angle theta from the reference direction,
    #         the two orbit equations give the same distance:
    #             p1 / (1 + e1 cos(theta - argp1)) = p2 / (1 + e2 cos(theta - argp2)).
    #         Cross-multiplying and expanding the cosines gives
    #             A cos(theta) + B sin(theta) = C,   with
    A = p1 * e2 * np.cos(argp2) - p2 * e1 * np.cos(argp1)
    B = p1 * e2 * np.sin(argp2) - p2 * e1 * np.sin(argp1)
    C = p2 - p1

    # Step 3: the left side can reach any value between -sqrt(A^2+B^2) and
    #         +sqrt(A^2+B^2), and no other. The small margin makes orbits
    #         that just touch count as intersecting despite rounding.
    reach = np.hypot(A, B)
    margin = tol * np.maximum(p1, p2)
    return out(np.abs(C) <= reach + margin)


# --- Coplanar transfers ------------------------------------------------------


def hohmann(r1, r2, mu):
    """Hohmann transfer between two coplanar circular orbits.

    Parameters
    ----------
    r1 : radius of the initial orbit
    r2 : radius of the final orbit (larger or smaller than r1)
    mu : gravitational parameter

    Returns
    -------
    dv1, dv2 : signed velocity changes of the first and second burn
    tof : transfer time, half the period of the transfer ellipse

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 6, coplanar transfers.

    Notes
    -----
    Both burns are prograde for an outward transfer (r2 > r1) and retrograde
    for an inward one. Total cost: ``abs(dv1) + abs(dv2)``.
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)

    # Step 1: the transfer ellipse touches both circles, so its apses are at
    #         r1 and r2 and its major axis is  2a = r1 + r2.
    a_transfer = 0.5 * (r1 + r2)

    # Step 2: first burn, at r1: from the circular speed to the speed of the
    #         transfer ellipse at that radius. Both velocities are tangential,
    #         so the velocity change is the difference of the speeds.
    dv1 = visviva_speed(r1, a_transfer, mu) - circular_speed(r1, mu)

    # Step 3: second burn, at r2: from the transfer ellipse to the circle.
    dv2 = circular_speed(r2, mu) - visviva_speed(r2, a_transfer, mu)

    # Step 4: the transfer covers half of the ellipse: half its period.
    tof = np.pi * np.sqrt(a_transfer**3 / mu)
    return dv1, dv2, tof


def bielliptic(r1, r2, rb, mu):
    """Bi-elliptic transfer between two coplanar circular orbits.

    Three tangential burns: at r1 onto a first ellipse with apoapsis rb; at rb
    onto a second ellipse with periapsis r2; at r2 to circularise.

    Parameters
    ----------
    r1 : radius of the initial orbit
    r2 : radius of the final orbit
    rb : radius of the intermediate apoapsis (rb >= max(r1, r2))
    mu : gravitational parameter

    Returns
    -------
    dv1, dv2, dv3 : signed velocity changes of the three burns
    tof : transfer time, the sum of the two half periods

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 6, coplanar transfers.

    Notes
    -----
    For an outward transfer the third burn is retrograde (negative). With
    ``rb = r2`` the velocity changes reduce to those of the Hohmann transfer
    (``dv3 = 0``); the returned time is then longer than the Hohmann time by
    half a revolution of the final orbit.
    """
    r1, r2, rb = np.asarray(r1, float), np.asarray(r2, float), np.asarray(rb, float)
    # Step 1: two half-ellipses. The first has apses r1 and rb, the second
    #         rb and r2.
    a_first = 0.5 * (r1 + rb)
    a_second = 0.5 * (r2 + rb)

    # Step 2: first burn at r1, circle -> first ellipse.
    dv1 = visviva_speed(r1, a_first, mu) - circular_speed(r1, mu)

    # Step 3: second burn at rb, first ellipse -> second ellipse. Far away
    #         the spacecraft is slow, so this burn is small.
    dv2 = visviva_speed(rb, a_second, mu) - visviva_speed(rb, a_first, mu)

    # Step 4: third burn at r2, second ellipse -> circle. Arriving from
    #         outside the spacecraft is too fast: the burn is retrograde.
    dv3 = circular_speed(r2, mu) - visviva_speed(r2, a_second, mu)

    # Step 5: time = half the period of each ellipse.
    tof = np.pi * (np.sqrt(a_first**3 / mu) + np.sqrt(a_second**3 / mu))
    return dv1, dv2, dv3, tof


def hohmann_vs_bielliptic(radius_ratio, rb_ratio):
    """Total cost of the Hohmann and of the bi-elliptic transfer, made dimensionless.

    Parameters
    ----------
    radius_ratio : r2 / r1, final over initial radius
    rb_ratio : rb / r1, intermediate apoapsis over initial radius

    Returns
    -------
    cost_hohmann, cost_bielliptic : total velocity change of each transfer,
        divided by the circular speed of the initial orbit

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 6, the bi-elliptic transfer: when it wins.

    Notes
    -----
    For a radius ratio below about 11.94 the Hohmann transfer is always
    cheaper; above about 15.58 any bi-elliptic transfer with ``rb > r2`` is
    cheaper; in between it depends on ``rb``.
    """
    # Canonical units of the initial orbit: r1 = 1 and mu = 1, so the initial
    # circular speed is 1 and every velocity change comes out already divided
    # by it.
    dv1, dv2, _ = hohmann(1.0, radius_ratio, 1.0)
    db1, db2, db3, _ = bielliptic(1.0, radius_ratio, rb_ratio, 1.0)
    cost_hohmann = np.abs(dv1) + np.abs(dv2)
    cost_bielliptic = np.abs(db1) + np.abs(db2) + np.abs(db3)
    return cost_hohmann, cost_bielliptic


# --- Burn of finite duration --------------------------------------------------


def finite_burn(r0_N, v0_N, dv, burn_time, mu, rtol=1e-11):
    """State at the end of a burn of finite duration, thrust along the velocity.

    The impulsive approximation replaces a burn by a jump of the velocity.
    This function integrates the burn itself: the thrust is in the equation
    of motion, r'' = -mu r / |r|^3 + a_T v / |v|, with a constant thrust
    acceleration a_T = dv / burn_time. Comparing its result with
    ``apply_impulse`` shows what the impulsive approximation neglects.

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,)
        Position and velocity at ignition, components in N. One state only.
    dv : float
        Velocity change that the engine delivers, a_T times the burn time.
        Positive along the velocity, negative against it.
    burn_time : float
        Duration of the burn, greater than zero.
    mu : float
        Gravitational parameter.
    rtol : float
        Relative tolerance of the integration (method DOP853 of SciPy).

    Returns
    -------
    r_N, v_N : ndarray, shape (3,)
        Position and velocity at cutoff, components in N.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 6, equation of motion with thrust; finite burn or impulse.

    Notes
    -----
    The mass is taken as constant during the burn, so that the thrust
    acceleration is constant; a real engine of constant thrust accelerates
    more towards the end. The thrust follows the direction of the velocity
    during the burn. For ``burn_time`` much shorter than the orbital period
    the result tends to the impulsive one.
    """
    from scipy.integrate import solve_ivp

    r0_N = np.asarray(r0_N, dtype=float)
    v0_N = np.asarray(v0_N, dtype=float)
    thrust_acceleration = dv / burn_time                 # constant magnitude, signed

    def derivative(t, x_N):
        r_N, v_N = x_N[:3], x_N[3:]
        gravity_N = -mu * r_N / np.linalg.norm(r_N) ** 3
        thrust_N = thrust_acceleration * v_N / np.linalg.norm(v_N)   # along the velocity
        return np.concatenate([v_N, gravity_N + thrust_N])

    x0_N = np.concatenate([r0_N, v0_N])
    atol = 1e-13 * np.concatenate([np.full(3, np.linalg.norm(r0_N)), np.full(3, np.linalg.norm(v0_N))])
    solution = solve_ivp(derivative, (0.0, burn_time), x0_N, method="DOP853", rtol=rtol, atol=atol)
    return solution.y[:3, -1], solution.y[3:, -1]


__all__ = [
    "exhaust_velocity", "rocket_dv", "mass_ratio", "propellant_mass",
    "apply_impulse", "energy_change", "apse_change_dv", "single_impulse_dv",
    "orbits_intersect", "hohmann", "bielliptic", "hohmann_vs_bielliptic",
    "finite_burn",
]
