import numpy as np
from scipy.optimize import fsolve
from flightdynamics import calculate_derivatives


def trim_residuals(unknowns, V):
    alpha, delta_e, T = unknowns
    gamma = 0.0
    theta = alpha
    q = 0.0
    V_dot, gamma_dot, _, q_dot = calculate_derivatives(V, gamma, theta, q, delta_e, T)
    return [V_dot, gamma_dot, q_dot]


def trim(V):
    """
    Trim for straight and level flight at airspeed V.

    Returns:
        x0 : trim state  [V, gamma, theta, q]
        u0 : trim inputs [delta_e, T]
    """
    initial_guess = [np.deg2rad(3.0), 0.0, 1000.0]
    solution, info, ok, msg = fsolve(trim_residuals, initial_guess, args=(V,), full_output=True)

    if ok != 1:
        raise RuntimeError(f"Trim failed at V = {V} m/s: {msg}")

    alpha, delta_e, T = solution
    x0 = np.array([V, 0.0, alpha, 0.0])
    u0 = np.array([delta_e, T])
    return x0, u0


if __name__ == "__main__":
    for V in [30.0, 40.0, 50.0, 60.0]:
        x0, u0 = trim(V)
        residuals = calculate_derivatives(*x0, *u0)
        print(
            f"V = {V:4.0f} m/s | "
            f"alpha = {np.rad2deg(x0[2]):6.2f} deg | "
            f"delta_e = {np.rad2deg(u0[0]):6.2f} deg | "
            f"T = {u0[1]:7.1f} N | "
            f"max residual = {np.max(np.abs(residuals)):.1e}"
        )