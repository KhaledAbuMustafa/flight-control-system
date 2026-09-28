"""
6-DOF rigid-body flight dynamics (body axes, flat earth, constant air density).

State  x = [u, v, w, p, q, r, phi, theta, psi, x_N, y_E, h]   (12)
Input  c = [delta_e, delta_a, delta_r, T]                      (4)

Aerodynamics (stage 1 = same terms as the 3-DOF model, plus the full lateral set):
    CL = CL0 + CL_alpha*alpha                      (CL_delta_e, CL_q, CL_alpha_dot follow later)
    CD = CD0 + k*CL^2
    Cm = Cm0 + Cm_alpha*alpha + Cm_delta_e*de + Cm_q*q_hat + Cm_alpha_dot*alpha_dot_hat
    CY, Cl, Cn = full lateral-directional model

INCLUDE_ALPHA_DOT = True  -> Cm_alpha_dot term active (set False to reproduce the 3-DOF model)
"""
import numpy as np
from aircraft import p

# State indices (makes the code readable)
U, V_, W, P, Q, R, PHI, THETA, PSI, XN, YE, H = range(12)
DE, DA, DR, TH = range(4)

INCLUDE_ALPHA_DOT = True      # pitch damping from the rate of change of alpha (Cm_alpha_dot)


def air_data(x, wind_ned=(0.0, 0.0, 0.0)):
    """Airspeed V, angle of attack alpha, sideslip beta from body velocities (minus wind)."""
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    wind_body = rotation_body_to_ned(phi, theta, psi).T @ np.asarray(wind_ned)
    u, v, w = x[U:W + 1] - wind_body                 # velocity relative to the air
    V = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / V)
    return V, alpha, beta


def rotation_body_to_ned(phi, theta, psi):
    """Direction cosine matrix body -> NED for the 3-2-1 (psi, theta, phi) Euler sequence."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
        [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
        [-sth,      sph * cth,                   cph * cth],
    ])


def aero_forces_moments(x, c, wind_ned=(0.0, 0.0, 0.0), alpha_dot=0.0):
    """Aerodynamic forces (body axes) and moments [N, N·m]. alpha_dot in rad/s."""
    V, alpha, beta = air_data(x, wind_ned)
    p_, q_, r_ = x[P], x[Q], x[R]
    de, da, dr = c[DE], c[DA], c[DR]

    q_bar = 0.5 * p.rho * V**2
    p_hat = p_ * p.b / (2 * V)                       # rates made dimensionless
    q_hat = q_ * p.c_bar / (2 * V)
    alpha_dot_hat = alpha_dot * p.c_bar / (2 * V)    # made dimensionless like q
    r_hat = r_ * p.b / (2 * V)

    # Longitudinal coefficients (identical to the 3-DOF model)
    CL = p.CL0 + p.CL_alpha * alpha
    CD = p.CD0 + p.k * CL**2
    Cm = (p.Cm0 + p.Cm_alpha * alpha + p.Cm_delta_e * de + p.Cm_q * q_hat
          + p.Cm_alpha_dot * alpha_dot_hat)

    # Lateral-directional coefficients
    CY = p.CY_beta * beta + p.CY_delta_a * da + p.CY_delta_r * dr + p.CY_p * p_hat + p.CY_r * r_hat
    Cl = p.Cl_beta * beta + p.Cl_delta_a * da + p.Cl_delta_r * dr + p.Cl_p * p_hat + p.Cl_r * r_hat
    Cn = p.Cn_beta * beta + p.Cn_delta_a * da + p.Cn_delta_r * dr + p.Cn_p * p_hat + p.Cn_r * r_hat

    # Lift/drag (wind axes) -> body axes (small-beta approximation)
    X = q_bar * p.S * (CL * np.sin(alpha) - CD * np.cos(alpha))
    Y = q_bar * p.S * CY
    Z = q_bar * p.S * (-CL * np.cos(alpha) - CD * np.sin(alpha))

    L_roll = q_bar * p.S * p.b * Cl
    M = q_bar * p.S * p.c_bar * Cm
    N = q_bar * p.S * p.b * Cn

    # Moment transfer to the actual CG: the aerodynamic forces act at the reference
    # point, which lies dx = dh_cg * c_bar AHEAD of the CG when the CG moves aft.
    dx = p.dh_cg * p.c_bar
    M += -dx * Z                                     # lift ahead of CG -> nose-up moment
    N += dx * Y                                      # side force ahead of CG -> weaker weathercock
    return np.array([X, Y, Z]), np.array([L_roll, M, N])


def f(x, c, wind_ned=(0.0, 0.0, 0.0)):
    """x_dot = f(x, c): the four equation blocks."""
    u, v, w, p_, q_, r_, phi, theta, psi = x[:9]
    (X, Y, Z), _ = aero_forces_moments(x, c, wind_ned)   # forces do not depend on alpha_dot
    X += c[TH]                                       # thrust along the body x-axis

    g, m = p.g, p.m
    sph, cph = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)

    # Block 1: forces
    u_dot = r_ * v - q_ * w + X / m - g * sth
    v_dot = p_ * w - r_ * u + Y / m + g * sph * cth
    w_dot = q_ * u - p_ * v + Z / m + g * cph * cth

    # alpha_dot from the translational accelerations: alpha = atan(w/u)
    #   -> alpha_dot = (u*w_dot - w*u_dot) / (u^2 + w^2)
    # then the moments are evaluated with this alpha_dot (Cm_alpha_dot term)
    alpha_dot = (u * w_dot - w * u_dot) / (u**2 + w**2) if INCLUDE_ALPHA_DOT else 0.0
    _, (L_roll, M, N) = aero_forces_moments(x, c, wind_ned, alpha_dot)

    # Block 2: moments (p_dot and r_dot are coupled through I_xz -> 2x2 system)
    Ix, Iy, Iz, Ixz = p.I_x, p.I_y, p.I_z, p.I_xz
    rhs_roll = L_roll + (Iy - Iz) * q_ * r_ + Ixz * p_ * q_
    rhs_yaw = N + (Ix - Iy) * p_ * q_ - Ixz * q_ * r_
    p_dot, r_dot = np.linalg.solve([[Ix, -Ixz], [-Ixz, Iz]], [rhs_roll, rhs_yaw])
    q_dot = (M + (Iz - Ix) * p_ * r_ + Ixz * (r_**2 - p_**2)) / Iy

    # Block 3: Euler angles
    phi_dot = p_ + (q_ * sph + r_ * cph) * np.tan(theta)
    theta_dot = q_ * cph - r_ * sph
    psi_dot = (q_ * sph + r_ * cph) / cth

    # Block 4: position (NED velocity, height = -z_down)
    vel_ned = rotation_body_to_ned(phi, theta, psi) @ np.array([u, v, w])
    xN_dot, yE_dot, h_dot = vel_ned[0], vel_ned[1], -vel_ned[2]

    return np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot,
                     phi_dot, theta_dot, psi_dot, xN_dot, yE_dot, h_dot])


def rk4_step(x, c, dt, wind_ned=(0.0, 0.0, 0.0)):
    """One Runge-Kutta 4th-order step (more accurate than Euler for the same dt)."""
    k1 = f(x, c, wind_ned)
    k2 = f(x + 0.5 * dt * k1, c, wind_ned)
    k3 = f(x + 0.5 * dt * k2, c, wind_ned)
    k4 = f(x + dt * k3, c, wind_ned)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def state_from_3dof(V, gamma, theta, q, h=1000.0):
    """Build a 6-DOF state from the longitudinal 3-DOF state (wings level, no sideslip)."""
    alpha = theta - gamma
    x = np.zeros(12)
    x[U], x[W] = V * np.cos(alpha), V * np.sin(alpha)
    x[Q], x[THETA], x[H] = q, theta, h
    return x