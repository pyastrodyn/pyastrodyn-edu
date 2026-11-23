"""The Keplerian restricted two-body problem: equation of motion and integrals.

Layer 2 (numerical). Any consistent unit system. Vectors are given by their
components in an inertial basis N, with shape (3,) for one state or (n, 3)
for n states.

NUMERICAL INTEGRATION, IN BRIEF
-------------------------------
The equation of motion  r'' = -mu r / |r|^3  is of second order. Integrators
want first order, so we stack position and velocity into one state vector

    x = [ r ; v ]   (6 numbers),      x' = [ v ; -mu r / |r|^3 ].

Integrating this Cartesian form directly is known as COWELL'S METHOD. Two
integrators are offered:

- ``propagate_numerical``: an adaptive integrator from SciPy (default DOP853,
  a Dormand-Prince Runge-Kutta method of order 8). It chooses its own step
  size so that the estimated error of each step stays below a tolerance.
- ``propagate_rk4``: the classical fixed-step Runge-Kutta method of order 4,
  written out in full so that every operation can be read.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ._util import col, dot, norm


def twobody_eom(t, x_N, mu):
    """Right-hand side of the two-body equation of motion, in first-order form.

    Parameters
    ----------
    t : float
        Time. Not used (the system is autonomous); present because numerical
        integrators expect the signature ``f(t, x)``.
    x_N : ndarray, shape (6,) or (n, 6)
        State: position then velocity, components in N.
    mu : float
        Gravitational parameter.

    Returns
    -------
    xdot_N : ndarray, same shape as ``x_N``
        Velocity, then acceleration ``-mu r / |r|^3``.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, hypotheses and equation of motion.
    """
    x_N = np.asarray(x_N, dtype=float)

    # Step 1: split the state. "[..., :3]" takes the first three numbers of
    #         every state, "[..., 3:]" the last three (primer 4).
    r_N = x_N[..., :3]                           # position, shape (3,) or (n, 3)
    v_N = x_N[..., 3:]                           # velocity

    # Step 2: acceleration of gravity,  a = -mu r / |r|^3.
    #         The scalar mu/|r|^3 multiplies a vector, hence col() (primer 3).
    a_N = -col(mu / norm(r_N) ** 3) * r_N

    # Step 3: the derivative of [r ; v] is [v ; a]. Join them on the last axis.
    return np.concatenate((v_N, a_N), axis=-1)


def propagate_numerical(r0_N, v0_N, t_eval, mu, rtol: float = 1e-11, atol=None,
                        method: str = "DOP853"):
    """Numerical integration of the two-body equation of motion (adaptive step).

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,)
        Initial position and velocity at time 0 (one trajectory per call).
    t_eval : ndarray, shape (m,)
        Increasing, non-negative times at which the state is returned.
    mu : float
        Gravitational parameter.
    rtol : float
        Relative tolerance per step. Default 1e-11.
    atol : float, ndarray of shape (6,), or None
        Absolute tolerance per step. Default (None): 1e-13 times the initial
        distance for the position components and 1e-13 times the initial
        speed for the velocity components.
    method : str
        Integration method of SciPy's ``solve_ivp``. Default "DOP853";
        "RK45" (Dormand-Prince, order 5) is a common lower-order choice.

    Returns
    -------
    r_N, v_N : ndarray, shape (m, 3)
        Position and velocity at the requested times.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, hypotheses and equation of motion.

    Notes
    -----
    How the accuracy is controlled. At every step the integrator estimates
    its own error, component by component, and accepts the step only if

        error <= atol + rtol * |value of that component|.

    Otherwise it repeats the step with a smaller step size. Three points:

    - SciPy's own defaults (rtol = 1e-3, atol = 1e-6) are far too loose for
      orbits: on the test orbit below they leave an error of several
      kilometres after ONE revolution with this method, and of tens of
      kilometres with SciPy's default method, RK45. The defaults here are
      much tighter.
    - The default ``atol`` is tied to the size of the initial state, so the
      result is the same in km and s as in canonical units.
    - The tolerances bound the error of EACH STEP, not the error at the end,
      which accumulates along the trajectory. Measured against the exact
      Kepler solution on an Earth orbit with a = 11000 km and e = 0.35, the
      position error with the defaults is about 1 mm after 1.5 revolutions,
      0.2 m after 10 and 25 m after 100: it grows roughly with the square of
      the number of revolutions.

    ``rtol`` cannot usefully be set below about 1e-13 (the limit of double
    precision).
    """
    r0_N = np.asarray(r0_N, dtype=float)
    v0_N = np.asarray(v0_N, dtype=float)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    if r0_N.shape != (3,) or v0_N.shape != (3,):
        raise ValueError("propagate_numerical takes one initial state: shapes (3,) and (3,).")

    # Step 1: initial state vector  x0 = [r0 ; v0], six numbers.
    x0_N = np.concatenate((r0_N, v0_N))

    # Step 2: absolute tolerance, one value per component of the state.
    #         Default: scaled to the problem, so it means the same in any units.
    if atol is None:
        atol_position = 1e-13 * norm(r0_N)
        atol_velocity = 1e-13 * norm(v0_N)
        atol = np.array([atol_position] * 3 + [atol_velocity] * 3)

    # Step 3: integrate from time 0 to the last requested time. SciPy calls
    #         twobody_eom(t, x, mu) as often as it needs and reports the
    #         state at the times in t_eval.
    solution = solve_ivp(
        twobody_eom, (0.0, t_eval[-1]), x0_N,
        method=method, t_eval=t_eval, args=(mu,), rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Numerical integration failed: {solution.message}")

    # Step 4: SciPy returns solution.y with shape (6, m): one COLUMN per time.
    #         We want one ROW per time: take the first three and the last
    #         three rows and transpose each to shape (m, 3).
    r_N = solution.y[:3].T
    v_N = solution.y[3:].T
    return r_N, v_N


def rk4_step(x_N, h, mu):
    """One step of the classical Runge-Kutta method of order 4 for the two-body problem.

    Parameters
    ----------
    x_N : ndarray, shape (6,) or (n, 6)
        State (position then velocity) at the start of the step.
    h : float
        Step size (time).
    mu : float
        Gravitational parameter.

    Returns
    -------
    ndarray, same shape as ``x_N``
        State at the end of the step.

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, hypotheses and equation of motion.

    Notes
    -----
    The error of one step is proportional to h^5, and the error accumulated
    over a fixed time span to h^4: halving the step divides the error by
    about sixteen. The time does not appear because the equation of motion
    does not depend on it.
    """
    x_N = np.asarray(x_N, dtype=float)
    # Four evaluations of the slope x' = f(x):
    k1 = twobody_eom(0.0, x_N, mu)                    # at the start
    k2 = twobody_eom(0.0, x_N + 0.5 * h * k1, mu)     # at the midpoint, using k1
    k3 = twobody_eom(0.0, x_N + 0.5 * h * k2, mu)     # at the midpoint, using k2
    k4 = twobody_eom(0.0, x_N + h * k3, mu)           # at the end, using k3
    # Weighted average of the four slopes (weights 1, 2, 2, 1 over 6).
    return x_N + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_rk4(r0_N, v0_N, h, n_steps: int, mu):
    """Fixed-step integration of the two-body problem with ``rk4_step``.

    A teaching integrator: every operation is visible, and the effect of the
    step size on the accuracy can be measured directly. For accurate work use
    ``propagate_numerical``.

    Parameters
    ----------
    r0_N, v0_N : ndarray, shape (3,)
        Initial position and velocity at time 0.
    h : float
        Step size.
    n_steps : int
        Number of steps; the final time is ``n_steps * h``.
    mu : float
        Gravitational parameter.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Times, from 0 to ``n_steps * h``.
    r_N, v_N : ndarray, shape (n_steps + 1, 3)
        Position and velocity at those times (row 0 is the initial state).

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, hypotheses and equation of motion.

    Notes
    -----
    No error control: the accuracy depends entirely on the step size chosen
    by the user. As a guide, measured with a step of 1/100 of the orbital
    period: the position error after one revolution is a few millionths of
    the semi-major axis for e up to 0.2, but about a thousand times larger
    for e = 0.6. A fixed step wastes effort far from the focus and is too
    coarse near periapsis, which is why adaptive integrators exist.
    """
    r0_N = np.asarray(r0_N, dtype=float)
    v0_N = np.asarray(v0_N, dtype=float)

    # Step 1: room for the whole history: one row per time, six numbers each.
    history = np.zeros((n_steps + 1, 6))
    history[0] = np.concatenate((r0_N, v0_N))

    # Step 2: march forward, one step at a time.
    for k in range(n_steps):
        history[k + 1] = rk4_step(history[k], h, mu)

    # Step 3: times, and the history split into position and velocity.
    t = h * np.arange(n_steps + 1)
    return t, history[:, :3], history[:, 3:]


def angular_momentum(r_N, v_N):
    """Specific angular momentum vector, h = r x v.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)

    Returns
    -------
    h_N : ndarray, same shape

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 1, conservation of the angular momentum and planar motion.
    """
    # np.cross works on the last axis: one cross product per state.
    return np.cross(np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float))


def eccentricity_vector(r_N, v_N, mu):
    """Eccentricity vector: points to the periapsis, magnitude e.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
    mu : float

    Returns
    -------
    e_N : ndarray, same shape
        ``((v^2 - mu/r) r - (r . v) v) / mu``

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, conservation of the eccentricity vector.
    """
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)
    r = norm(r_N)                                # distance
    v_squared = dot(v_N, v_N)                    # speed squared
    r_dot_v = dot(r_N, v_N)                      # zero at the apses
    # Two scalars each multiply a vector, hence col() (primer 3).
    return (col(v_squared - mu / r) * r_N - col(r_dot_v) * v_N) / col(mu)


def specific_energy(r_N, v_N, mu):
    """Specific mechanical energy, v^2/2 - mu/r.

    Parameters
    ----------
    r_N, v_N : ndarray, shape (3,) or (n, 3)
    mu : float

    Returns
    -------
    float or ndarray, shape (n,)

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*.
    Lecture 1, energy.
    """
    kinetic = 0.5 * dot(v_N, v_N)                # per unit mass
    potential = -mu / norm(r_N)                  # per unit mass
    return kinetic + potential
def gravitational_acceleration(r, mu):
    """Magnitude of the gravitational acceleration at a distance from the centre of attraction.

    Parameters
    ----------
    r : float or ndarray
        Distance from the centre of the attracting body.
    mu : float
        Gravitational parameter.

    Returns
    -------
    g = mu / r^2, directed toward the centre.

    References
    ----------
    Curtis, *Orbital Mechanics for Engineering Students*; Vallado,
    *Fundamentals of Astrodynamics and Applications*.
    Lecture 1, gravitational acceleration decay.

    Notes
    -----
    Inverse-square law: at twice the distance the acceleration is one fourth.
    Valid outside a body with spherical symmetry.
    """
    return np.asarray(mu, float) / np.asarray(r, float) ** 2


__all__ = [
    "twobody_eom", "propagate_numerical", "rk4_step", "propagate_rk4",
    "angular_momentum", "eccentricity_vector", "specific_energy", "gravitational_acceleration",
]
