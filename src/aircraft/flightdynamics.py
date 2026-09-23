import numpy as np
import c172_params as p


# ============================================================
# Aerodynamics
# ============================================================

def lift_coefficient(alpha):
    return p.CL0 + p.CL_alpha * alpha


def drag_coefficient(CL):
    return p.CD0 + p.k * CL**2


def pitching_moment_coefficient(alpha, delta_e, q, V):
    q_hat = q * p.c_bar / (2.0 * V)          # dimensionless pitch rate
    return (
        p.Cm0
        + p.Cm_alpha * alpha
        + p.Cm_delta_e * delta_e
        + p.Cm_q * q_hat                     # pitch damping
    )


def dynamic_pressure(V):
    return 0.5 * p.rho * V**2


# ============================================================
# Flight dynamics
# ============================================================

def calculate_derivatives(V, gamma, theta, q, delta_e, T):

    # Angle of attack
    alpha = theta - gamma

    # Aerodynamic coefficients
    CL = lift_coefficient(alpha)
    CD = drag_coefficient(CL)
    Cm = pitching_moment_coefficient(alpha, delta_e, q, V)

    # Aerodynamic forces and moment
    q_dyn = dynamic_pressure(V)
    L = q_dyn * p.S * CL
    D = q_dyn * p.S * CD
    M = q_dyn * p.S * p.c_bar * Cm

    # Equations of motion
    V_dot = (T * np.cos(alpha) - D - p.m * p.g * np.sin(gamma)) / p.m
    gamma_dot = (L + T * np.sin(alpha) - p.m * p.g * np.cos(gamma)) / (p.m * V)
    theta_dot = q
    q_dot = M / p.I_y

    return V_dot, gamma_dot, theta_dot, q_dot